# -*- coding: utf-8 -*-
"""
backtest_tv_events_mp.py
- TradingView signals (Paul indicator events) 백테스트
- 멀티프로세싱 지원, distance_pct 필터 지원
"""

import os
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import Tuple, List, Dict

from sr_engine.data import get_ohlcv
from sr_engine.simulate import simulate_event_group


# -------------------- 유틸 --------------------

def ts_to_ns(s: pd.Series) -> np.ndarray:
    """UTC aware datetime 시리즈를 ns int64 배열로 변환"""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.view("int64").to_numpy()


def filter_by_distance(df: pd.DataFrame, dist_max: float) -> pd.DataFrame:
    """distance_pct 필터 적용"""
    if dist_max is None:
        return df
    before = len(df)
    df2 = df[df["distance_pct"].astype(float) <= dist_max * 100.0].copy()
    after = len(df2)
    print(f"[BT] distance_pct filter {dist_max}: {before}->{after} rows")
    return df2


# -------------------- 시뮬레이션 --------------------

def simulate_symbol(symbol: str, rows: pd.DataFrame,
                    timeframe: str, tp: float, sl: float, fee: float,
                    expiries_h: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """한 심볼에 대한 시뮬"""
    ohlcv = get_ohlcv(symbol, timeframe)
    if ohlcv is None or len(ohlcv) == 0:
        print(f"[{symbol}] get_ohlcv returned empty.")
        return pd.DataFrame(), pd.DataFrame()

    trades, stats = simulate_event_group(
        symbol=symbol,
        ohlcv=ohlcv,
        rows=rows,
        tp_pct=tp,
        sl_pct=sl,
        fee_rt=fee,
        expiries_h=expiries_h
    )
    return trades, stats


def run_group(df: pd.DataFrame, group: str, timeframe: str,
              tp: float, sl: float, fee: float,
              expiries_h: List[float], procs: int):

    df_g = df[df["event"] == group].copy()
    rows = len(df_g)
    syms = df_g["symbol"].nunique()
    print(f"[BT][{group}] start: symbols={syms} rows={rows} tasks={syms} procs={procs}")

    if rows == 0 or syms == 0:
        print(f"[BT][{group}] no tasks.")
        return pd.DataFrame(), pd.DataFrame()

    tasks = [(sym, df_g[df_g["symbol"] == sym], timeframe, tp, sl, fee, expiries_h)
             for sym in df_g["symbol"].unique()]

    if procs > 1:
        with mp.Pool(processes=procs) as pool:
            parts = pool.starmap(simulate_symbol, tasks)
    else:
        parts = [simulate_symbol(*t) for t in tasks]

    trades = pd.concat([p[0] for p in parts if p[0] is not None and not p[0].empty],
                       ignore_index=True) if parts else pd.DataFrame()
    stats = pd.concat([p[1] for p in parts if p[1] is not None and not p[1].empty],
                      ignore_index=True) if parts else pd.DataFrame()

    if not trades.empty:
        print(f"[BT][{group}] trades -> rows={len(trades)}")
    else:
        print(f"[BT][{group}] no trades.")

    return trades, stats


# -------------------- 메인 --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv_enriched.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h", help="만료 시간 콤마구분 (예: 0.5h,1h,2h)")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dist-max", type=float, default=None, help="distance_pct 필터 (0.02 → 2%)")
    ap.add_argument("--outdir", default="./logs")
    args = ap.parse_args()

    # 입력
    df = pd.read_csv(args.signals)
    print(f"[BT] signals rows={len(df)}, symbols={df['symbol'].nunique()}, timeframe={args.timeframe}")

    # distance_pct 필터
    if args.dist_max is not None and "distance_pct" in df.columns:
        df = filter_by_distance(df, args.dist_max)

    # expiries 파싱
    expiries_h = []
    for e in args.expiries.split(","):
        e = e.strip().lower()
        if e.endswith("h"):
            expiries_h.append(float(e[:-1]))
        elif e.endswith("m"):
            expiries_h.append(float(e[:-1]) / 60.0)
        elif e.endswith("d"):
            expiries_h.append(float(e[:-1]) * 24.0)
    expiries_h = [float(x) for x in expiries_h]

    groups = ["detected", "price_in_box", "box_breakout", "line_breakout"]

    all_trades = []
    all_stats = []
    for grp in groups:
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl,
                           args.fee, expiries_h, args.procs)
        if not tr.empty:
            all_trades.append(tr)
        if not st.empty:
            all_stats.append(st)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    # 결과 저장
    os.makedirs(args.outdir, exist_ok=True)
    trades_path = os.path.join(args.outdir, "bt_tv_events_trades.csv")
    stats_path = os.path.join(args.outdir, "bt_tv_events_stats.csv")

    trades.to_csv(trades_path, index=False)
    stats.to_csv(stats_path, index=False)

    print(f"[BT] saved -> {trades_path} (rows={len(trades)})")
    print(f"[BT] saved -> {stats_path} (rows={len(stats)})")


if __name__ == "__main__":
    mp.freeze_support()
    main()