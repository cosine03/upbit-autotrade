# -*- coding: utf-8 -*-
"""
estimate_tv_levels_mp.py
- TradingView 알람 로그(signals_tv.csv)를 기반으로 레벨(가격대) 근사 추정 (멀티프로세싱 버전).
- OHLCV 로컬 CSV를 읽어 swings/클러스터 기반으로 추정 후 signals_tv_enriched.csv 생성.

기능:
  * multiprocessing.Pool + Ctrl+C 안전 종료 처리
  * tz-aware timestamp → int 변환 시 view() → astype()으로 경고 제거
  * lookback/lookahead 범위 지정 가능
  * outdir CSV 저장

실행 예시:
  python estimate_tv_levels_mp.py ./logs/signals_tv.csv `
    --timeframe 15m `
    --lookback 400 --lookahead 40 `
    --procs 12 --chunksize 128 `
    --out ./logs/signals_tv_enriched.csv
"""

import os, re, time, sys, argparse, signal
import multiprocessing as mp
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# -------------------- 공통 유틸 --------------------

def to_utc_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def ensure_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("OHLCV must have DatetimeIndex or 'ts' column")
        ts = df.index
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    out = df.copy()
    out["ts"] = ts
    return out.reset_index(drop=True)

def series_to_ns_utc(s: pd.Series) -> np.ndarray:
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.astype("int64").to_numpy()

def parse_side(row: pd.Series) -> Optional[str]:
    side = str(row.get("side", "") or "").strip().lower()
    if side in ("support", "resistance"):
        return side
    msg = str(row.get("message", "") or "")
    if re.search(r"resistance", msg, re.I):
        return "resistance"
    if re.search(r"support", msg, re.I):
        return "support"
    return None

# -------------------- 레벨 추정 로직 (단일행) --------------------

def estimate_level_for_signal(row: dict, lookback: int, lookahead: int, timeframe: str) -> dict:
    """
    하나의 알람 row → 레벨 추정 결과 딕셔너리
    """
    import sr_engine.data as data
    import sr_engine.levels as levels

    sym = str(row["symbol"]).strip()
    ts  = to_utc_ts(row["ts"])
    side = parse_side(row)

    # 1) OHLCV 로드
    df = data.get_ohlcv(sym, timeframe)
    if df is None or len(df) == 0:
        return {**row, "est_level": np.nan, "est_band": np.nan,
                "est_touches": np.nan, "distance_pct": np.nan,
                "method": "failed", "side_used": side or "", "sig_price": np.nan}

    df = ensure_ts_col(df)
    ts64 = series_to_ns_utc(df["ts"])
    key64 = np.int64(ts.value)
    i = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i < 0:
        return {**row, "method": "failed"}

    sig_price = float(df["close"].iloc[i])
    i0 = max(0, i - lookback)
    i1 = min(len(df) - 1, i + lookahead)
    seg = df.iloc[i0:i1+1].reset_index(drop=True)
    if seg.empty or len(seg) < 5:
        return {**row, "method": "failed"}

    band = float(levels.auto_deviation_band(seg))
    try:
        swings_raw = levels.find_swings(seg)
    except TypeError:
        swings_raw = levels.find_swings(seg, window=5)

    def to_idx(x):
        arr = np.array(x or [], dtype=int)
        return arr[(arr >= 0) & (arr < len(seg))]

    low_idx = to_idx(getattr(swings_raw, "low_idx", [])) if hasattr(swings_raw, "low_idx") else to_idx(swings_raw[0])
    high_idx = to_idx(getattr(swings_raw, "high_idx", [])) if hasattr(swings_raw, "high_idx") else to_idx(swings_raw[1])

    def cluster(idxs):
        if idxs.size == 0: return []
        pts = np.array([(i, float(seg["close"].iloc[i])) for i in idxs], dtype=float)
        pts = pts[np.argsort(pts[:,1])]
        clusters, cur_c, cur_m = [], pts[0,1], [int(pts[0,0])]
        for j in range(1, len(pts)):
            ii, px = int(pts[j,0]), pts[j,1]
            if abs(px - cur_c) <= band:
                cur_m.append(ii)
                cur_c = (cur_c*(len(cur_m)-1)+px)/len(cur_m)
            else:
                clusters.append({"center": cur_c, "idx": np.array(cur_m), "touches": len(cur_m)})
                cur_c, cur_m = px, [ii]
        clusters.append({"center": cur_c, "idx": np.array(cur_m), "touches": len(cur_m)})
        return clusters

    sup_levels = cluster(low_idx)
    res_levels = cluster(high_idx)

    cands = []
    if side == "support": cands.extend(("support", lv) for lv in sup_levels)
    elif side == "resistance": cands.extend(("resistance", lv) for lv in res_levels)
    else:
        cands.extend(("support", lv) for lv in sup_levels)
        cands.extend(("resistance", lv) for lv in res_levels)

    if not cands: return {**row, "method": "failed"}

    best, best_dist = None, None
    for sside, lv in cands:
        center = float(lv["center"])
        dist = abs(sig_price - center)/max(1e-9, sig_price)
        if best is None or dist < best_dist:
            best, best_dist = (sside, lv), dist

    side_sel, lv_sel = best
    return {**row,
            "est_level": float(lv_sel["center"]),
            "est_band": band,
            "est_touches": int(lv_sel.get("touches",0)),
            "distance_pct": best_dist*100.0,
            "method": f"swings_cluster[{lookback}/{lookahead}]",
            "side_used": side_sel,
            "sig_price": sig_price}

# -------------------- 멀티프로세싱 Pool 안전 실행 --------------------

def _init_worker():
    try: signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception: pass

def run_pool_map(func, tasks, procs: int, chunksize: int = 64):
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=procs, initializer=_init_worker)
    try:
        for res in pool.imap_unordered(func, tasks, chunksize=chunksize):
            yield res
    except KeyboardInterrupt:
        print("[EST] ^C detected → terminating workers...", flush=True)
        pool.terminate(); pool.join(); sys.exit(130)
    except Exception as e:
        print(f"[EST] ERROR in pool: {e!r}", flush=True)
        pool.terminate(); pool.join(); raise
    else:
        pool.close(); pool.join()

# -------------------- 메인 --------------------

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--lookahead", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.signals)
    if "ts" not in df.columns: raise ValueError("signals must have ts col")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "timeframe" not in df.columns: df["timeframe"] = args.timeframe
    df["timeframe"] = df["timeframe"].fillna(args.timeframe).astype(str)
    if "symbol" not in df.columns: raise ValueError("signals must have symbol col")
    if args.limit > 0: df = df.iloc[:args.limit]

    rows = df.to_dict("records")
    total = len(rows)
    print(f"[EST] start MP: rows={total}, timeframe(default)={args.timeframe}, look={args.lookback}/{args.lookahead}, procs={args.procs}, chunksize={args.chunksize}")

    worker = lambda row: estimate_level_for_signal(row, args.lookback, args.lookahead, row.get("timeframe", args.timeframe))

    outputs = []
    k = 0
    for out in run_pool_map(worker, rows, procs=args.procs, chunksize=args.chunksize):
        outputs.append(out); k += 1
        if k % 50 == 0: print(f"[EST] processed {k}/{total} ...", flush=True)

    out_df = pd.DataFrame(outputs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[EST] done. saved -> {args.out}")
    print(f"[EST] success {out_df['method'].ne('failed').sum()}/{len(out_df)} rows")

if __name__ == "__main__":
    main()