# -*- coding: utf-8 -*-
"""
backtest_tv_entry_close_mp.py (조건 완화 + 안정화 버전)

- TradingView signals 기반 백테스트
- Entry 조건 완화: signal_price 이상(= 저가 <= signal_price) 충족 시 진입
- 만기: 4h / 8h
- Long-only / Fees 0.1% roundtrip
- Multiprocessing 지원
"""

import os
import argparse
import pandas as pd
import numpy as np
import pyupbit
from datetime import timedelta
from multiprocessing import Pool, cpu_count

FEE = 0.001  # 왕복 수수료 0.1%

# -------------------------------
# OHLCV Loader
# -------------------------------
def get_ohlcv(sym, timeframe="15m", count=2000):
    try:
        df = pyupbit.get_ohlcv(sym, interval=timeframe, count=count)
        if df is None or df.empty:
            print(f"[{sym}] OHLCV load error: None or empty")
            return None
        df = df.reset_index()
        df.rename(columns={"index": "ts"}, inplace=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    except Exception as e:
        print(f"[{sym}] OHLCV load exception: {e}")
        return None

# -------------------------------
# Entry & Exit Simulation
# -------------------------------
def simulate_symbol(task):
    sym, rows, timeframe, tp, sl, expiry_h = task
    df = get_ohlcv(sym, timeframe)
    if df is None:
        return []

    trades = []
    for _, s in rows.iterrows():
        try:
            sig_ts = pd.Timestamp(s["ts"], tz="UTC")
            signal_price = float(s.get("level", s.get("close", np.nan)))
            if np.isnan(signal_price):
                continue

            # 시작 index
            ts = df["ts"].to_numpy()
            idx0 = np.searchsorted(ts, sig_ts.to_datetime64(), side="right")
            if idx0 >= len(df):
                continue

            expiry_ts = sig_ts + timedelta(hours=expiry_h)
            idx_exp = np.searchsorted(ts, expiry_ts.to_datetime64(), side="left")
            if idx_exp >= len(df):
                idx_exp = len(df) - 1

            # Entry 조건 완화: expiry 전까지 저가 <= signal_price 충족 시 진입
            entry_idx, entry_price = None, None
            for i in range(idx0, idx_exp + 1):
                low_i, close_i = df.loc[i, ["low", "close"]]
                if low_i <= signal_price:
                    entry_idx = i
                    entry_price = close_i  # 충족 캔들의 종가로 진입
                    break
            if entry_price is None:
                continue

            # Exit 탐색
            tp_price = entry_price * (1 + tp / 100)
            sl_price = entry_price * (1 - sl / 100)
            exit_price, exit_idx = None, None

            for j in range(entry_idx + 1, idx_exp + 1):
                high_j, low_j, close_j = df.loc[j, ["high", "low", "close"]]
                if high_j >= tp_price:
                    exit_price, exit_idx = tp_price, j
                    break
                if low_j <= sl_price:
                    exit_price, exit_idx = sl_price, j
                    break

            if exit_price is None:
                exit_price, exit_idx = df.loc[idx_exp, "close"], idx_exp

            net = (exit_price / entry_price - 1) - FEE
            trades.append({
                "symbol": sym,
                "ts": sig_ts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "tp": tp,
                "sl": sl,
                "expiry_h": expiry_h,
                "net": net,
            })
        except Exception as e:
            print(f"[{sym}] simulate error: {e}")
            continue

    return trades

# -------------------------------
# Aggregation
# -------------------------------
def agg_stats(trades):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    stats = (
        df.groupby(["tp", "sl", "expiry_h"], as_index=False)
        .agg(
            trades=("net", "count"),
            win_rate=("net", lambda x: (x > 0).mean()),
            avg_net=("net", "mean"),
            median_net=("net", "median"),
            total_net=("net", "sum"),
        )
    )
    return df, stats

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("signals", help="signals_tv.csv path")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--expiry", default="4h,8h")
    parser.add_argument("--procs", type=int, default=cpu_count() // 2)
    args = parser.parse_args()

    df_sig = pd.read_csv(args.signals)
    df_sig["ts"] = pd.to_datetime(df_sig["ts"], utc=True)

    EXPIRIES = [int(x.replace("h", "")) for x in args.expiry.split(",")]
    STRATS = [
        ("stable", 1.5, 1.0),
        ("aggressive", 2.0, 1.25),
        ("scalp", 1.0, 0.75),
        ("mid", 1.25, 1.0),
        ("mid2", 1.75, 1.25),
        ("tight", 0.8, 0.8),
    ]

    tasks = []
    for (sym, g) in df_sig.groupby("symbol"):
        for expiry_h in EXPIRIES:
            for name, tp, sl in STRATS:
                tasks.append((sym, g, args.timeframe, tp, sl, expiry_h))

    print(f"[BT] starting with {len(tasks)} tasks using {args.procs} procs...")
    trades_all = []
    with Pool(processes=args.procs) as pool:
        try:
            results = pool.map(simulate_symbol, tasks)
            for r in results:
                trades_all.extend(r)
        except KeyboardInterrupt:
            pool.terminate()
            print("[BT] interrupted by user")
            return

    trades_df, stats_df = agg_stats(trades_all)
    trades_df.to_csv("./logs/bt_tv_entryclose_trades.csv", index=False)
    stats_df.to_csv("./logs/bt_tv_entryclose_stats.csv", index=False)

    print("\n=== TV Backtest (Entry relaxed / Long-only / 0.1% fees) ===")
    print(f"Trades saved: ./logs/bt_tv_entryclose_trades.csv (rows={len(trades_df)})")
    print(f"Stats  saved: ./logs/bt_tv_entryclose_stats.csv (rows={len(stats_df)})")
    print(stats_df)

if __name__ == "__main__":
    main()