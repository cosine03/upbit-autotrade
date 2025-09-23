# -*- coding: utf-8 -*-
"""
backtest_tv_entry_close_mp.py
멀티프로세싱 버전 (TV 시그널)
- Entry 조건: 알람 발생 이후 가격이 알람 발생시점 가격 이하일 때 진입
- Exit 조건: TP / SL / Expiry (4h, 8h)
- Long only, 0.1% 왕복 수수료
"""

import os
import argparse
import multiprocessing as mp
from datetime import timedelta

import numpy as np
import pandas as pd
import pyupbit


# ===================== 데이터 로딩 =====================
def get_ohlcv(sym: str, timeframe: str = "15m", count: int = 4000) -> pd.DataFrame:
    df = pyupbit.get_ohlcv(sym, interval=timeframe, count=count)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.reset_index()
    df = df.rename(columns={"index": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# ===================== 헬퍼 =====================
def next_bar_open_idx(df: pd.DataFrame, sig_ts: pd.Timestamp) -> int:
    ts = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    sig_ts_naive = pd.Timestamp(sig_ts).tz_convert("UTC").tz_localize(None)
    return int(np.searchsorted(ts, np.datetime64(sig_ts_naive), side="right"))


# ===================== 시뮬레이션 =====================
def simulate_symbol(task):
    sym, rows, timeframe, tp, sl, expiry_h = task
    ohlcv = get_ohlcv(sym, timeframe)
    if ohlcv.empty:
        return pd.DataFrame()

    trades = []
    for _, s in rows.iterrows():
        sig_ts = pd.Timestamp(s["ts"]).tz_convert("UTC")
        sig_price = float(s.get("price", np.nan) or s.get("level", np.nan) or 0)
        if sig_price <= 0:
            continue

        try:
            i0 = next_bar_open_idx(ohlcv, sig_ts) - 1
        except Exception:
            continue
        if i0 < 0 or i0 >= len(ohlcv):
            continue

        # expiry 계산
        dt_expiry = sig_ts + timedelta(hours=expiry_h)
        dt_expiry_naive = dt_expiry.tz_convert("UTC").tz_localize(None)

        ts_naive = ohlcv["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        idx_exp = int(np.searchsorted(ts_naive, np.datetime64(dt_expiry_naive), side="left"))
        if idx_exp >= len(ohlcv):
            idx_exp = len(ohlcv) - 1

        entry_price = None
        entry_idx = None
        for i in range(i0, idx_exp + 1):
            bar = ohlcv.iloc[i]
            if float(bar["low"]) <= sig_price:
                entry_price = float(bar["open"])
                entry_idx = i
                break

        if entry_price is None:
            continue

        tp_price = entry_price * (1 + tp / 100)
        sl_price = entry_price * (1 + sl / 100)
        exit_price = None
        exit_ts = None
        exit_reason = None

        for i in range(entry_idx, idx_exp + 1):
            bar = ohlcv.iloc[i]
            if float(bar["high"]) >= tp_price:
                exit_price = tp_price
                exit_ts = bar["ts"]
                exit_reason = "tp"
                break
            if float(bar["low"]) <= sl_price:
                exit_price = sl_price
                exit_ts = bar["ts"]
                exit_reason = "sl"
                break
        if exit_price is None:
            bar = ohlcv.iloc[idx_exp]
            exit_price = float(bar["close"])
            exit_ts = bar["ts"]
            exit_reason = "expiry"

        fee = 0.001
        net = (exit_price / entry_price) - 1 - fee

        trades.append({
            "symbol": sym,
            "sig_ts": sig_ts,
            "sig_price": sig_price,
            "entry_ts": ohlcv.iloc[entry_idx]["ts"],
            "entry_price": entry_price,
            "exit_ts": exit_ts,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "tp": tp,
            "sl": sl,
            "expiry_h": expiry_h,
            "net": net
        })

    return pd.DataFrame(trades)


# ===================== 전략 정의 =====================
STRATS = [
    ("stable_1.5/1.0", 1.5, -1.0),
    ("aggressive_2.0/1.25", 2.0, -1.25),
    ("scalp_1.0/0.75", 1.0, -0.75),
    ("mid_1.25/1.0", 1.25, -1.0),
    ("mid_1.75/1.25", 1.75, -1.25),
    ("tight_0.8/0.8", 0.8, -0.8),
]


# ===================== 메인 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("signals", help="signals_tv.csv")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--expiry", default="4h,8h")
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--group", choices=["all", "major", "alt"], default="all")
    args = parser.parse_args()

    df = pd.read_csv(args.signals)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    if args.group == "major":
        majors = {"KRW-BTC", "KRW-ETH"}
        df = df[df["symbol"].isin(majors)]
    elif args.group == "alt":
        majors = {"KRW-BTC", "KRW-ETH"}
        df = df[~df["symbol"].isin(majors)]

    expiries = [int(e[:-1]) for e in args.expiry.split(",")]

    tasks = []
    for name, tp, sl in STRATS:
        for exp in expiries:
            strat_name = f"{name}_{exp}h"
            for sym, rows in df.groupby("symbol"):
                tasks.append((sym, rows, args.timeframe, tp, sl, exp))

    with mp.Pool(processes=args.procs) as pool:
        parts = pool.map(simulate_symbol, tasks)

    trades = pd.concat(parts, ignore_index=True)
    trades.to_csv("./logs/bt_tv_entryclose_trades.csv", index=False)

    def agg_stats(x):
        return pd.Series({
            "trades": len(x),
            "win_rate": (x["net"] > 0).mean(),
            "avg_net": x["net"].mean(),
            "median_net": x["net"].median(),
            "total_net": x["net"].sum(),
        })

    stats = trades.groupby(["tp", "sl", "expiry_h"], as_index=False).apply(agg_stats)
    stats.to_csv("./logs/bt_tv_entryclose_stats.csv", index=False)

    print("\n=== TV Backtest (Entry = price <= signal_price(1m) / Long-only / fees=0.1% RT) ===")
    print(f"Trades saved: ./logs/bt_tv_entryclose_trades.csv (rows={len(trades)})")
    print(f"Stats  saved: ./logs/bt_tv_entryclose_stats.csv (rows={len(stats)})")
    print(stats)


if __name__ == "__main__":
    main()