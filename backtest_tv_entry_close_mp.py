# -*- coding: utf-8 -*-
"""
backtest_tv_entry_close_mp.py

멀티프로세싱 버전 - TradingView 기반 TV 신호 백테스트
- Entry: signal 발생 이후 가격이 signal_price 이하가 되는 시점 진입
- Exit : 지정된 TP/SL or expiry (종가청산)
- Fees : 왕복 0.1%
- Long only
- Multiprocessing 지원 (--procs)

실행 예시:
  python backtest_tv_entry_close_mp.py ./logs/signals_tv.csv --group alt --expiry 4h,8h --procs 20
"""

import os, argparse, multiprocessing as mp
import pandas as pd, numpy as np
import pyupbit
from datetime import timedelta

# ====== 유틸 ======
def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise RuntimeError("signals file must have 'ts' column")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "symbol", "event"])
    return df

def get_ohlcv(symbol: str, tf: str, count: int = 2000) -> pd.DataFrame:
    df = pyupbit.get_ohlcv(ticker=symbol, interval=tf, count=count)
    df = df.reset_index().rename(columns={"index": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df

def next_bar_open_idx(df: pd.DataFrame, sig_ts: pd.Timestamp) -> int:
    ts = df["ts"].to_numpy()
    sig_ts_utc = sig_ts.tz_convert("UTC") if sig_ts.tzinfo else sig_ts.tz_localize("UTC")
    return int(np.searchsorted(ts, sig_ts_utc.to_datetime64(), side="right"))

# ====== 시뮬레이션 (개별 심볼) ======
def simulate_symbol(task):
    sym, rows, timeframe, tp, sl, expiry_h = task
    trades = []
    try:
        ohlcv = get_ohlcv(sym, timeframe)
    except Exception as e:
        print(f"[{sym}] OHLCV load error: {e}")
        return pd.DataFrame(columns=[
            "symbol","sig_ts","sig_price","entry_ts","entry_price",
            "exit_ts","exit_price","exit_reason","tp","sl","expiry_h","net"
        ])

    for _, s in rows.iterrows():
        sig_ts = pd.Timestamp(s["ts"], tz="UTC")
        sig_price = float(s.get("level", s.get("sig_price", np.nan)))
        if np.isnan(sig_price):
            continue

        i0 = next_bar_open_idx(ohlcv, sig_ts) - 1
        if i0 < 0 or i0 >= len(ohlcv) - 1:
            continue

        expiry_ts = sig_ts + timedelta(hours=expiry_h)
        idx_exp = int(np.searchsorted(
            ohlcv["ts"].to_numpy(),
            pd.Timestamp(expiry_ts).to_datetime64(),
            side="left"
        ))
        if idx_exp <= i0:
            continue

        entry_ts, entry_price = None, None
        for i in range(i0 + 1, idx_exp):
            if float(ohlcv["low"].iloc[i]) <= sig_price:
                entry_ts = ohlcv["ts"].iloc[i]
                entry_price = sig_price
                break
        if entry_price is None:
            continue

        tp_price = entry_price * (1 + tp/100)
        sl_price = entry_price * (1 - sl/100)
        exit_ts, exit_price, exit_reason = None, None, None

        for j in range(i, idx_exp):
            low_j = float(ohlcv["low"].iloc[j])
            high_j = float(ohlcv["high"].iloc[j])
            close_j = float(ohlcv["close"].iloc[j])
            ts_j = ohlcv["ts"].iloc[j]

            if high_j >= tp_price:
                exit_ts, exit_price, exit_reason = ts_j, tp_price, "tp"
                break
            if low_j <= sl_price:
                exit_ts, exit_price, exit_reason = ts_j, sl_price, "sl"
                break
        if exit_ts is None:
            exit_ts = ohlcv["ts"].iloc[min(idx_exp, len(ohlcv)-1)]
            exit_price = float(ohlcv.loc[ohlcv["ts"] == exit_ts, "close"])
            exit_reason = "expiry"

        fee = 0.001  # 왕복 0.1%
        net = (exit_price / entry_price - 1) - fee

        trades.append({
            "symbol": sym, "sig_ts": sig_ts, "sig_price": sig_price,
            "entry_ts": entry_ts, "entry_price": entry_price,
            "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": exit_reason,
            "tp": tp, "sl": sl, "expiry_h": expiry_h, "net": net
        })

    if not trades:
        return pd.DataFrame(columns=[
            "symbol","sig_ts","sig_price","entry_ts","entry_price",
            "exit_ts","exit_price","exit_reason","tp","sl","expiry_h","net"
        ])
    return pd.DataFrame(trades)

# ====== 통계 ======
def agg_stats(df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "trades": len(df),
        "win_rate": (df["net"] > 0).mean() if len(df) else 0,
        "avg_net": df["net"].mean() if len(df) else 0,
        "median_net": df["net"].median() if len(df) else 0,
        "total_net": df["net"].sum() if len(df) else 0,
    })

# ====== 메인 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="4h,8h")
    ap.add_argument("--group", default="all", choices=["all","major","alt"])
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    df_sig = load_signals(args.signals)
    majors = ["KRW-BTC","KRW-ETH"]
    if args.group == "major":
        df_sig = df_sig[df_sig["symbol"].isin(majors)]
    elif args.group == "alt":
        df_sig = df_sig[~df_sig["symbol"].isin(majors)]

    EXPIRIES = [int(x.replace("h","")) for x in args.expiry.split(",")]
    STRATS = [
        ("stable", 1.5, 1.0),
        ("aggressive", 2.0, 1.25),
        ("scalp", 1.0, 0.75),
        ("mid", 1.25, 1.0),
        ("mid2", 1.75, 1.25),
        ("tight", 0.8, 0.8),
    ]

    tasks = []
    for sym, rows in df_sig.groupby("symbol"):
        for nm,tp,sl in STRATS:
            for expiry_h in EXPIRIES:
                tasks.append((sym, rows, args.timeframe, tp, sl, expiry_h))

    with mp.Pool(processes=args.procs) as pool:
        parts = pool.map(simulate_symbol, tasks)

    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[
        "symbol","sig_ts","sig_price","entry_ts","entry_price",
        "exit_ts","exit_price","exit_reason","tp","sl","expiry_h","net"
    ])

    stats = trades.groupby(["tp","sl","expiry_h"], as_index=False).apply(agg_stats)

    out_trades = "./logs/bt_tv_entryclose_trades.csv"
    out_stats  = "./logs/bt_tv_entryclose_stats.csv"
    os.makedirs("./logs", exist_ok=True)
    trades.to_csv(out_trades, index=False)
    stats.to_csv(out_stats, index=False)

    print("\n=== TV Backtest (Entry=price<=signal_price / Long-only / fees=0.1%) ===")
    print(f"Trades saved: {out_trades} (rows={len(trades)})")
    print(f"Stats  saved: {out_stats} (rows={len(stats)})")
    print(stats)

if __name__ == "__main__":
    main()