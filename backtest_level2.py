import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

import pyupbit


def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    return df


def get_ohlcv(symbol: str, interval: str, count: int = 400) -> pd.DataFrame:
    """업비트에서 OHLCV 데이터 다운로드"""
    df = pyupbit.get_ohlcv(symbol, interval=interval, count=count)
    if df is None:
        return None
    df.index = df.index.tz_localize("UTC")
    return df


def backtest(df_signals, interval, stop, take, expire_days):
    results = []
    for _, sig in df_signals.iterrows():
        if sig["event"] != "price_in_box":
            continue
        if sig["side"] != "support" or sig["level"] != 2:
            continue

        sym = sig["symbol"]
        ts = sig["ts_utc"]

        ohlcv = get_ohlcv(sym, interval=interval, count=400)
        if ohlcv is None or ts not in ohlcv.index:
            continue

        # 진입은 다음 캔들의 시가
        try:
            entry_idx = ohlcv.index.get_loc(ts) + 1
        except KeyError:
            continue
        if entry_idx >= len(ohlcv):
            continue

        entry_time = ohlcv.index[entry_idx]
        entry_price = ohlcv.iloc[entry_idx]["open"]

        stop_price = entry_price * (1.0 + stop)
        take_price = entry_price * (1.0 + take)
        expire_time = entry_time + timedelta(days=expire_days)

        trade_outcome = None
        exit_time = None
        exit_price = None

        for t, row in ohlcv.iloc[entry_idx + 1:].iterrows():
            if t > expire_time:
                trade_outcome = "expire"
                exit_time = expire_time
                exit_price = row["close"]
                break
            if row["low"] <= stop_price:
                trade_outcome = "stop"
                exit_time = t
                exit_price = stop_price
                break
            if row["high"] >= take_price:
                trade_outcome = "take"
                exit_time = t
                exit_price = take_price
                break

        if trade_outcome is None:
            trade_outcome = "open"
            exit_time = ohlcv.index[-1]
            exit_price = ohlcv.iloc[-1]["close"]

        results.append({
            "symbol": sym,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "outcome": trade_outcome,
            "R": (exit_price - entry_price) / entry_price
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", default="./logs/signals.csv")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--stop", type=float, default=-0.02)
    parser.add_argument("--take", type=float, default=0.04)
    parser.add_argument("--expire-days", type=int, default=2)
    args = parser.parse_args()

    df_signals = load_signals(args.signals)

    df_trades = backtest(
        df_signals,
        interval=args.timeframe,
        stop=args.stop,
        take=args.take,
        expire_days=args.expire_days,
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trades_path = f"./logs/level2_trades_{ts}.csv"
    summary_path = f"./logs/level2_summary_{ts}.csv"

    os.makedirs("logs", exist_ok=True)
    df_trades.to_csv(trades_path, index=False)

    summary = df_trades.groupby("outcome")["R"].agg(["count", "mean", "median"])
    summary.to_csv(summary_path)

    print("=== SUMMARY ===")
    print(summary)
    print("\nSaved:")
    print(" ", trades_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
