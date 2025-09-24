# -*- coding: utf-8 -*-
"""
estimate_tv_levels_mp.py
- TV 알람 시점 레벨 근사 추정 (멀티프로세싱, Windows 호환)
- 경고 제거: Series.view -> astype 경로로 변경 (tz-aware 안전)
"""

import os, re, time, argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from multiprocessing import Pool, freeze_support, cpu_count

# 프로젝트 의존
from sr_engine.data import get_ohlcv
from sr_engine.levels import auto_deviation_band, find_swings

# ---------- 유틸 ----------
def to_utc_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def ensure_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        ts = df.index
        ts = ts.tz_localize("UTC") if getattr(ts, "tz", None) is None else ts.tz_convert("UTC")
    out = df.copy()
    out["ts"] = ts
    return out.reset_index(drop=True)

def series_to_ns_utc(s: pd.Series) -> np.ndarray:
    """tz-aware → tz-naive 변환 후 int64(ns)로 안전 변환 (pandas 2.2+ 호환)"""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    # tz-aware면 tz 제거
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    # 명시적으로 datetime64[ns] → int64(ns)
    return s.astype("datetime64[ns]").astype("int64").to_numpy()

def parse_side(row: dict) -> str:
    side = str(row.get("side", "") or "").strip().lower()
    if side in ("support", "resistance"):
        return side
    msg = str(row.get("message", "") or "")
    if re.search(r"resistance", msg, re.I): return "resistance"
    if re.search(r"support", msg, re.I):    return "support"
    return ""

# ---------- 레벨 추정 ----------
def estimate_level_for_signal(symbol, sig_ts, side_hint, timeframe, lookback=400, lookahead=40):
    df = get_ohlcv(symbol, timeframe)
    if df is None or len(df) == 0:
        return None
    df = ensure_ts_col(df)
    ts = df["ts"]

    sig_ts = to_utc_ts(sig_ts)
    ts64 = series_to_ns_utc(ts)
    key64 = np.int64(sig_ts.value)  # ns
    i = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i < 0: return None
    sig_price = float(df["close"].iloc[i])

    i0 = max(0, i - lookback)
    i1 = min(len(df) - 1, i + lookahead)
    seg = df.iloc[i0:i1+1].reset_index(drop=True)
    if seg.empty or len(seg) < 5:
        return None

    band = float(auto_deviation_band(seg))
    try:
        swings_raw = find_swings(seg)
    except TypeError:
        swings_raw = find_swings(seg, window=5)

    def norm_idx(x):
        if x is None: return np.array([], dtype=int)
        arr = pd.to_numeric(x, errors="coerce").to_numpy()
        arr = arr[np.isfinite(arr)].astype(int, copy=False)
        return np.clip(arr, 0, len(seg)-1)

    low_idx  = norm_idx(getattr(swings_raw, "get", lambda k: [])("low_idx"))
    high_idx = norm_idx(getattr(swings_raw, "get", lambda k: [])("high_idx"))

    def cluster(idxs):
        if idxs.size == 0: return []
        prices = seg["close"].to_numpy()
        pts = np.array([(ii, float(prices[ii])) for ii in idxs if 0 <= ii < len(prices)], dtype=float)
        if pts.size == 0: return []
        pts = pts[np.argsort(pts[:,1])]
        clusters, cur_c, cur_m = [], pts[0,1], [int(pts[0,0])]
        for j in range(1, len(pts)):
            ii, px = int(pts[j,0]), pts[j,1]
            if abs(px - cur_c) <= band:
                cur_m.append(ii)
                cur_c = (cur_c*(len(cur_m)-1) + px)/len(cur_m)
            else:
                clusters.append({"center": float(cur_c), "touches": len(cur_m)})
                cur_c, cur_m = px, [ii]
        clusters.append({"center": float(cur_c), "touches": len(cur_m)})
        return clusters

    sup_levels = cluster(low_idx)
    res_levels = cluster(high_idx)
    cands = []
    if side_hint == "support":
        cands += [("support", lv) for lv in sup_levels]
    elif side_hint == "resistance":
        cands += [("resistance", lv) for lv in res_levels]
    else:
        cands += [("support", lv) for lv in sup_levels]
        cands += [("resistance", lv) for lv in res_levels]
    if not cands: return None

    best, best_dist = None, None
    for side, lv in cands:
        center = float(lv["center"])
        dist = abs(sig_price - center) / max(1e-9, sig_price)
        if best is None or dist < best_dist:
            best, best_dist = (side, lv), dist

    side_sel, lv_sel = best
    return {
        "est_level": float(lv_sel["center"]),
        "est_band": float(band),
        "est_touches": int(lv_sel.get("touches", 0)),
        "distance_pct": float(best_dist * 100.0),
        "method": f"swings_cluster[{lookback}/{lookahead}]",
        "side_used": side_sel,
        "sig_price": float(sig_price),
    }

# ---------- 워커 ----------
def worker(task: dict) -> dict:
    try:
        est = estimate_level_for_signal(
            symbol   = task["symbol"],
            sig_ts   = task["ts"],
            side_hint= task["side"],
            timeframe= task["timeframe"],
            lookback = task["lookback"],
            lookahead= task["lookahead"],
        )
    except Exception:
        est = None

    out = dict(task["row"])
    if est: out.update(est)
    else:
        out.update({
            "est_level": np.nan, "est_band": np.nan, "est_touches": np.nan,
            "distance_pct": np.nan, "method": "failed", "side_used": task["side"] or "",
            "sig_price": np.nan,
        })
    return out

# ---------- 메인 ----------
def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "15m"))
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--lookahead", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=cpu_count())
    ap.add_argument("--chunksize", type=int, default=128)
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.signals)
    if "ts" not in df.columns:
        raise ValueError("signals 파일에 'ts' 컬럼 필요")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "timeframe" not in df.columns:
        df["timeframe"] = args.timeframe
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    tasks = []
    for _, row in df.iterrows():
        tasks.append({
            "row": row.to_dict(),
            "symbol": str(row["symbol"]),
            "ts": row["ts"],
            "side": parse_side(row.to_dict()),
            "timeframe": str(row.get("timeframe", args.timeframe)),
            "lookback": args.lookback,
            "lookahead": args.lookahead,
        })

    print(f"[EST] start MP: rows={len(tasks)}, timeframe(default)={args.timeframe}, "
          f"look={args.lookback}/{args.lookahead}, procs={args.procs}, chunksize={args.chunksize}")

    rows, t0 = [], time.time()
    with Pool(processes=args.procs) as pool:
        for out in pool.imap_unordered(worker, tasks, chunksize=args.chunksize):
            rows.append(out)
            if len(rows) % 50 == 0:
                print(f"[EST] processed {len(rows)}/{len(tasks)} ...")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    took = time.time() - t0
    ok = out_df["method"].ne("failed").sum()
    print(f"[EST] done. saved -> {args.out}")
    print(f"[EST] success {ok}/{len(out_df)} rows, took {took:.1f}s")

if __name__ == "__main__":
    freeze_support()
    main()