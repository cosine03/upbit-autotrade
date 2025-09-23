# -*- coding: utf-8 -*-
"""
backtest_tv_breakout_close_mp.py
- 단일 버전과 동일 로직을 멀티프로세싱으로 병렬 수행
- 옵션:
    signals: 경로 (필수)
    --timeframe  (기본 15m)
    --procs      (기본 6)
"""

import os
import sys
import argparse
from multiprocessing import get_context
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from sr_engine.data import get_ohlcv
from sr_engine.levels import auto_deviation_band, find_swings

FEE_RT = 0.001

# ---------- 공통 유틸 ----------
def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["ts"] = df.index
        else:
            raise RuntimeError("input has no 'ts'")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.reset_index(drop=True)

def tf_minutes(tf: str) -> int:
    s = tf.lower()
    if s.endswith("m"): return int(s[:-1])
    if s.endswith("h"): return int(s[:-1]) * 60
    if s.endswith("d"): return int(s[:-1]) * 60 * 24
    return 15

def next_bar_index_by_time(ts_array: np.ndarray, t: pd.Timestamp) -> int:
    t64 = np.datetime64(t.to_datetime64())
    return int(np.searchsorted(ts_array, t64, side="right"))

def cluster_by_price(df: pd.DataFrame, idxs: np.ndarray, band: float) -> List[Dict]:
    if idxs.size == 0:
        return []
    prices = df["close"].to_numpy()
    pts = np.array([(i, float(prices[i])) for i in idxs if 0 <= i < len(prices)], dtype=float)
    if pts.size == 0:
        return []
    pts = pts[np.argsort(pts[:,1])]
    clusters = []
    cur_c = pts[0,1]; cur_m = [int(pts[0,0])]
    for k in range(1, len(pts)):
        ii = int(pts[k,0]); px = pts[k,1]
        if abs(px - cur_c) <= band:
            cur_m.append(ii)
            cur_c = (cur_c * (len(cur_m)-1) + px) / len(cur_m)
        else:
            clusters.append({"center": float(cur_c), "idx": np.array(sorted(set(cur_m)), dtype=int)})
            cur_c = px; cur_m = [ii]
    clusters.append({"center": float(cur_c), "idx": np.array(sorted(set(cur_m)), dtype=int)})
    for c in clusters:
        c["touches"] = int(len(c["idx"]))
    return clusters

def normalize_swings(swings, n: int) -> Dict[str, np.ndarray]:
    def to_int_idx(x):
        if x is None: return np.array([], dtype=int)
        arr = np.array(x)
        if arr.dtype.kind not in ("i","u"):
            arr = pd.to_numeric(arr, errors="coerce").to_numpy()
        arr = arr[np.isfinite(arr)].astype(int, copy=False)
        if n: arr = np.clip(arr, 0, n-1)
        return np.unique(arr)
    if isinstance(swings, dict):
        return {"low_idx": to_int_idx(swings.get("low_idx")),
                "high_idx": to_int_idx(swings.get("high_idx"))}
    if isinstance(swings, (list, tuple)) and len(swings) >= 2:
        return {"low_idx": to_int_idx(swings[0]), "high_idx": to_int_idx(swings[1])}
    return {"low_idx": np.array([], dtype=int), "high_idx": np.array([], dtype=int)}

def pick_resistance_level(df_upto: pd.DataFrame, min_touches: int) -> Tuple[float,int]:
    band = float(auto_deviation_band(df_upto))
    try:
        sw_raw = find_swings(df_upto)
    except TypeError:
        sw_raw = find_swings(df_upto, window=5)
    sw = normalize_swings(sw_raw, len(df_upto))
    res_lv = cluster_by_price(df_upto, sw["high_idx"], band)
    res_lv = [lv for lv in res_lv if int(lv.get("touches",0)) >= min_touches]
    if not res_lv:
        return None, 0
    last_close = float(df_upto["close"].iloc[-1])
    centers = np.array([lv["center"] for lv in res_lv], dtype=float)
    pick = int(np.argmin(np.abs(centers - last_close)))
    return float(res_lv[pick]["center"]), int(res_lv[pick].get("touches",0))

# ---------- 워커 ----------
def worker(args_tuple):
    symbol, rows_dicts, timeframe = args_tuple
    rows = pd.DataFrame(rows_dicts)
    df = get_ohlcv(symbol, timeframe)
    if df is None or len(df) < 50:
        return pd.DataFrame()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index":"ts"})
        else:
            df = df.reset_index().rename(columns={df.index.name or "index":"ts"})
    df = ensure_ts(df)
    ts64 = df["ts"].to_numpy(dtype="datetime64[ns]")

    out = []
    for _, s in rows.iterrows():
        if str(s.get("side","")).lower() != "resistance":
            continue
        try:
            sig_ts = pd.Timestamp(s["ts"], tz="UTC") if pd.Timestamp(s["ts"]).tzinfo is None else pd.Timestamp(s["ts"]).tz_convert("UTC")
        except Exception:
            sig_ts = pd.to_datetime(s["ts"], utc=True)

        i_sig_next = next_bar_index_by_time(ts64, sig_ts)
        if i_sig_next >= len(df):
            continue
        i_upto = max(1, i_sig_next - 1)
        df_upto = df.iloc[:i_upto].copy()

        # line_close_24h
        cen2, t2 = pick_resistance_level(df_upto, 2)
        if cen2 is not None:
            c_next = float(df["close"].iloc[i_sig_next])
            if c_next > cen2:
                entry_ts = df["ts"].iloc[i_sig_next]
                exit_ts = entry_ts + pd.Timedelta(hours=24)
                idx_exit = int(np.searchsorted(ts64, np.datetime64(exit_ts), side="left"))
                if idx_exit < len(df):
                    ret = (float(df["close"].iloc[idx_exit]) / c_next) - 1.0 - FEE_RT
                    out.append({
                        "symbol": symbol, "strategy": "line_close_24h",
                        "touches": t2, "level_center": cen2,
                        "entry_ts": entry_ts.isoformat(), "entry_px": c_next,
                        "exit_ts": df["ts"].iloc[idx_exit].isoformat(), "exit_px": float(df["close"].iloc[idx_exit]),
                        "net": ret
                    })

        # box_close_24h
        cen3, t3 = pick_resistance_level(df_upto, 3)
        if cen3 is not None:
            c_next = float(df["close"].iloc[i_sig_next])
            if c_next > cen3:
                entry_ts = df["ts"].iloc[i_sig_next]
                exit_ts = entry_ts + pd.Timedelta(hours=24)
                idx_exit = int(np.searchsorted(ts64, np.datetime64(exit_ts), side="left"))
                if idx_exit < len(df):
                    ret = (float(df["close"].iloc[idx_exit]) / c_next) - 1.0 - FEE_RT
                    out.append({
                        "symbol": symbol, "strategy": "box_close_24h",
                        "touches": t3, "level_center": cen3,
                        "entry_ts": entry_ts.isoformat(), "entry_px": c_next,
                        "exit_ts": df["ts"].iloc[idx_exit].isoformat(), "exit_px": float(df["close"].iloc[idx_exit]),
                        "net": ret
                    })

    return pd.DataFrame(out)

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="path to signals_tv.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()

    df_sig = pd.read_csv(args.signals)
    for col in ("ts","event","side","symbol"):
        if col not in df_sig.columns:
            raise RuntimeError(f"signals file missing column: {col}")
    df_sig["ts"] = pd.to_datetime(df_sig["ts"], utc=True, errors="coerce")
    df_sig = df_sig.dropna(subset=["ts","symbol","side"]).copy()
    syms = sorted(df_sig["symbol"].str.upper().unique().tolist())

    tasks = []
    for sym in syms:
        rows = df_sig[df_sig["symbol"].str.upper() == sym][["ts","side","symbol"]].copy()
        tasks.append((sym, rows.to_dict("records"), args.timeframe))

    print(f"[BT] starting with {len(tasks)} symbols using {args.procs} procs...")
    ctx = get_context("spawn")
    with ctx.Pool(processes=args.procs) as pool:
        parts = pool.map(worker, tasks)

    trades = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame(
        columns=["symbol","strategy","touches","level_center","entry_ts","entry_px","exit_ts","exit_px","net"]
    )

    if not trades.empty:
        stats = (
            trades.groupby("strategy", as_index=False)
                  .agg(trades=("net","size"),
                       win_rate=("net", lambda x: float((x>0).mean())),
                       avg_net=("net","mean"),
                       median_net=("net","median"),
                       total_net=("net","sum"))
        )
    else:
        stats = pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"])

    os.makedirs("./logs", exist_ok=True)
    trades.to_csv("./logs/bt_tv_breakout_close_trades.csv", index=False)
    stats.to_csv("./logs/bt_tv_breakout_close_stats.csv", index=False)

    print("\n=== TV Breakout (close-based) Backtest / 24h expiry / Long-only / 0.1% RT ===")
    print(f"Trades saved: ./logs/bt_tv_breakout_close_trades.csv (rows={len(trades)})")
    print(f"Stats  saved: ./logs/bt_tv_breakout_close_stats.csv (rows={len(stats)})\n")
    if not stats.empty:
        print(stats.to_string(index=False))

if __name__ == "__main__":
    main()