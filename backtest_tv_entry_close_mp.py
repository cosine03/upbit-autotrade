# -*- coding: utf-8 -*-
"""
backtest_tv_entry_close_mp.py (multiprocessing + 안전판)

- TV 알람 기반 백테스트
- 진입: 알람 발생 시점 이후 가격이 signal_price 이하일 때
- 청산: expiry_h (예: 4h, 8h) 후 종가
- Long-only
- Multiprocessing 지원
- pyupbit.get_ohlcv() None 반환 및 tz 문제 대응
"""

import os, sys, argparse, multiprocessing
import pandas as pd
import numpy as np
import pyupbit

# ======================= 안전한 OHLCV 로드 =======================
def get_ohlcv_safe(symbol, tf, count=200):
    try:
        df = pyupbit.get_ohlcv(ticker=symbol, interval=tf, count=count)
        if df is None or len(df) == 0:
            print(f"[{symbol}] OHLCV is None/empty, skipping")
            return None
        df = df.reset_index()
        df["ts"] = pd.to_datetime(df["index"], utc=True)
        df = df.drop(columns=["index"])
        return df
    except Exception as e:
        print(f"[{symbol}] OHLCV load error: {e}")
        return None

# ======================= Timestamp 보정 =======================
def to_utc(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    else:
        return ts.tz_convert("UTC")

# ======================= 보조 함수 =======================
def next_bar_open_idx(ohlcv: pd.DataFrame, sig_ts: pd.Timestamp) -> int:
    ts = pd.to_datetime(ohlcv["ts"].to_numpy(), utc=True)
    sig_ts_utc = to_utc(sig_ts)
    return int(np.searchsorted(ts, sig_ts_utc.to_datetime64(), side="right"))

# ======================= 시뮬레이션 =======================
def simulate_symbol(task):
    sym, rows, timeframe, tp, sl, expiry_h = task
    ohlcv = get_ohlcv_safe(sym, timeframe, count=500)
    if ohlcv is None:
        return []

    out = []
    for _, s in rows.iterrows():
        try:
            sig_ts = to_utc(s["ts"])
            sig_price = float(s.get("price", np.nan))
            if np.isnan(sig_price):
                continue

            i0 = next_bar_open_idx(ohlcv, sig_ts) - 1
            if i0 < 0 or i0 >= len(ohlcv) - 1:
                continue

            # expiry 계산
            dt_expiry = sig_ts + pd.Timedelta(hours=expiry_h)
            ts_arr = ohlcv["ts"].to_numpy()
            idx_exp = int(np.searchsorted(ts_arr, np.datetime64(to_utc(dt_expiry)), side="left"))
            if idx_exp <= i0:
                continue

            dfw = ohlcv.iloc[i0:idx_exp+1].copy()

            # 진입 조건: 알람 이후 가격이 sig_price 이하
            df_after = dfw[dfw["ts"] > sig_ts]
            entry = df_after[df_after["low"] <= sig_price]
            if entry.empty:
                continue

            entry_idx = entry.index[0]
            entry_price = sig_price
            exit_price = float(dfw.iloc[-1]["close"])
            net = (exit_price / entry_price - 1.0) - 0.001  # 0.1% fee

            out.append({
                "symbol": sym,
                "ts": sig_ts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "net": net,
                "tp": tp,
                "sl": sl,
                "expiry_h": expiry_h,
            })
        except Exception as e:
            print(f"[{sym}] simulate error: {e}")
            continue

    return out

# ======================= 통계 =======================
def agg_stats(df):
    return pd.Series({
        "trades": len(df),
        "win_rate": (df["net"] > 0).mean() if len(df) else 0,
        "avg_net": df["net"].mean() if len(df) else 0,
        "median_net": df["net"].median() if len(df) else 0,
        "total_net": df["net"].sum() if len(df) else 0,
    })

# ======================= 메인 =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals_csv", help="signals_tv.csv 파일 경로")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="4h,8h", help="예: 4h,8h")
    ap.add_argument("--group", choices=["major","alt","all"], default="all")
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.signals_csv)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    majors = ["KRW-BTC","KRW-ETH"]
    if args.group == "major":
        df = df[df["symbol"].isin(majors)]
    elif args.group == "alt":
        df = df[~df["symbol"].isin(majors)]

    expiries = []
    for e in args.expiry.split(","):
        e = e.strip()
        if e.endswith("h"):
            expiries.append(int(e[:-1]))

    STRATS = [
        ("stable", 1.5, 1.0),
        ("aggressive", 2.0, 1.25),
        ("scalp", 1.0, 0.75),
        ("mid", 1.25, 1.0),
        ("mid", 1.75, 1.25),
        ("tight", 0.8, 0.8),
    ]

    tasks = []
    for sym, rows in df.groupby("symbol"):
        for strat, tp, sl in STRATS:
            for exp in expiries:
                tasks.append((sym, rows, args.timeframe, tp, sl, exp))

    print(f"[BT] starting with {len(tasks)} tasks using {args.procs} procs...")
    with multiprocessing.Pool(processes=args.procs) as pool:
        parts = pool.map(simulate_symbol, tasks)

    trades = pd.DataFrame([x for part in parts for x in part])
    if trades.empty:
        print("No trades generated.")
        return

    stats = trades.groupby(["tp","sl","expiry_h"], as_index=False).apply(agg_stats)

    out_trades = "./logs/bt_tv_entryclose_trades.csv"
    out_stats  = "./logs/bt_tv_entryclose_stats.csv"
    trades.to_csv(out_trades, index=False)
    stats.to_csv(out_stats, index=False)

    print(f"\n=== TV Backtest (Entry = price <= signal_price / Long-only / fees=0.1% RT) ===")
    print(f"Trades saved: {out_trades} (rows={len(trades)})")
    print(f"Stats  saved: {out_stats} (rows={len(stats)})")
    print(stats)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()