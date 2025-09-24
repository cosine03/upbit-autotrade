# -*- coding: utf-8 -*-
"""
estimate_tv_levels_mp.py  (MP, with local-CSV fallback + reason logs)
- TradingView(폴 지표) 알람 시점의 레벨(가격대)을 근사 추정해서 signals_tv.csv를 보강 (멀티프로세싱)
- 1번 패치: get_ohlcv 실패 시 로컬 CSV 자동 로딩, 실패 사유 상세 로그
- 2번 패치: pandas Series.view 경고 제거 (astype으로 대체)

사용 예:
  python -u estimate_tv_levels_mp.py ./logs/signals_tv.csv \
    --timeframe 15m --lookback 400 --lookahead 40 \
    --procs 12 --chunksize 128 \
    --out ./logs/signals_tv_enriched.csv

선택 옵션(없어도 됨):
  --assume-ohlcv-tz UTC
  --ohlcv-roots ".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv"
  --ohlcv-patterns "data/ohlcv/{symbol}-{timeframe}.csv;ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv"

주의:
- sr_engine이 설치되어 있으면 이를 사용, 없으면 로컬 CSV로만 처리
- 모든 ts는 UTC 기준
"""

import os
import re
import sys
import glob
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ---- sr_engine optional import ------------------------------------------------
SE_HAVE = True
try:
    from sr_engine.data import get_ohlcv as se_get_ohlcv  # type: ignore
    from sr_engine.levels import auto_deviation_band as se_auto_band, find_swings as se_find_swings  # type: ignore
except Exception:
    SE_HAVE = False
    se_get_ohlcv = None
    se_auto_band = None
    se_find_swings = None

# -------------------- 공통 유틸 --------------------

def to_utc_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def ensure_ts_col(df: pd.DataFrame, assume_tz: Optional[str] = None) -> pd.DataFrame:
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("OHLCV must have DatetimeIndex or 'ts' column")
        ts = df.index
        if ts.tz is None:
            if assume_tz:
                ts = ts.tz_localize(assume_tz).tz_convert("UTC")
            else:
                ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    out = df.copy()
    out["ts"] = ts
    return out.reset_index(drop=True)

def series_to_ns_utc(s: pd.Series) -> np.ndarray:
    # pandas 2.2 대응: view 대신 astype 사용
    s = pd.to_datetime(s, utc=True, errors="coerce")
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
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

# ---- Local CSV fallback loader ------------------------------------------------

def _default_roots() -> List[str]:
    return [".", "./data", "./data/ohlcv", "./ohlcv", "./logs", "./logs/ohlcv"]

def _default_patterns() -> List[str]:
    return [
        "data/ohlcv/{symbol}-{timeframe}.csv",
        "data/ohlcv/{symbol}_{timeframe}.csv",
        "ohlcv/{symbol}-{timeframe}.csv",
        "ohlcv/{symbol}_{timeframe}.csv",
        "logs/ohlcv/{symbol}-{timeframe}.csv",
        "logs/ohlcv/{symbol}_{timeframe}.csv",
        "{symbol}-{timeframe}.csv",
        "{symbol}_{timeframe}.csv",
    ]

def _find_ohlcv_csv(symbol: str, timeframe: str,
                    roots: Optional[List[str]] = None,
                    patterns: Optional[List[str]] = None) -> Optional[str]:
    roots = roots or _default_roots()
    patterns = patterns or _default_patterns()
    for r in roots:
        for pat in patterns:
            pat2 = pat.format(symbol=symbol, timeframe=timeframe)
            path = Path(r) / pat2 if not pat2.startswith((".", "/", "\\")) else Path(pat2)
            for p in glob.glob(str(path)):
                if Path(p).is_file():
                    return p
    return None

def _load_ohlcv_local(symbol: str, timeframe: str,
                      assume_tz: Optional[str],
                      roots: Optional[List[str]],
                      patterns: Optional[List[str]]) -> Optional[pd.DataFrame]:
    path = _find_ohlcv_csv(symbol, timeframe, roots, patterns)
    if not path:
        return None
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        def pick(*names):
            for n in names:
                if n in cols: return cols[n]
            return None
        c_ts = pick("ts","time","timestamp","datetime","date")
        c_o  = pick("open","o")
        c_h  = pick("high","h")
        c_l  = pick("low","l")
        c_c  = pick("close","c")
        c_v  = pick("volume","vol","v")
        need = [c_ts, c_o, c_h, c_l, c_c]
        if any(x is None for x in need):
            return None
        out = pd.DataFrame({
            "ts":    pd.to_datetime(df[c_ts], utc=True, errors="coerce"),
            "open":  pd.to_numeric(df[c_o], errors="coerce"),
            "high":  pd.to_numeric(df[c_h], errors="coerce"),
            "low":   pd.to_numeric(df[c_l], errors="coerce"),
            "close": pd.to_numeric(df[c_c], errors="coerce"),
        })
        if c_v is not None and c_v in df.columns:
            out["volume"] = pd.to_numeric(df[c_v], errors="coerce")
        out = out.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)
        return ensure_ts_col(out, assume_tz)
    except Exception:
        return None

# ---- Fallback swing/band when sr_engine not present ---------------------------

def _auto_deviation_band_fallback(seg: pd.DataFrame) -> float:
    # 간단한 대체: close의 이동표준편차를 사용 (너무 작지 않게 하한)
    c = seg["close"].to_numpy(dtype=float)
    if len(c) < 10:
        return max(1e-9, np.std(c) * 0.2)
    return max(1e-9, float(pd.Series(c).rolling(20, min_periods=10).std().dropna().median() or np.std(c)*0.2))

def _find_swings_fallback(seg: pd.DataFrame, window: int = 5) -> Dict[str, np.ndarray]:
    # 단순 로컬 extrema
    n = len(seg)
    lows, highs = [], []
    H = seg["high"].to_numpy(dtype=float)
    L = seg["low"].to_numpy(dtype=float)
    for i in range(window, n - window):
        if L[i] == np.min(L[i-window:i+window+1]):
            lows.append(i)
        if H[i] == np.max(H[i-window:i+window+1]):
            highs.append(i)
    return {"low_idx": np.array(lows, dtype=int), "high_idx": np.array(highs, dtype=int)}

def _auto_band(seg: pd.DataFrame) -> float:
    if se_auto_band is not None:
        try:
            return float(se_auto_band(seg))
        except Exception:
            pass
    return _auto_deviation_band_fallback(seg)

def _find_swings(seg: pd.DataFrame):
    if se_find_swings is not None:
        try:
            return se_find_swings(seg)
        except TypeError:
            try:
                return se_find_swings(seg, window=5)
            except Exception:
                return _find_swings_fallback(seg, window=5)
        except Exception:
            return _find_swings_fallback(seg, window=5)
    return _find_swings_fallback(seg, window=5)

def _normalize_swings(swings, n_rows: int) -> Dict[str, np.ndarray]:
    def to_int_idx(x):
        if x is None:
            return np.array([], dtype=int)
        arr = np.array(x)
        if arr.dtype.kind not in ("i","u"):
            arr = pd.to_numeric(arr, errors="coerce").to_numpy()
        arr = arr[np.isfinite(arr)].astype(int, copy=False)
        if n_rows:
            arr = np.clip(arr, 0, n_rows-1)
        return np.unique(arr)
    if isinstance(swings, dict):
        return {
            "low_idx": to_int_idx(swings.get("low_idx")),
            "high_idx": to_int_idx(swings.get("high_idx")),
        }
    if isinstance(swings, (list, tuple)) and len(swings) >= 2:
        return {"low_idx": to_int_idx(swings[0]), "high_idx": to_int_idx(swings[1])}
    return {"low_idx": np.array([], dtype=int), "high_idx": np.array([], dtype=int)}

def _cluster_by_price(df: pd.DataFrame, idxs: np.ndarray, band: float) -> List[Dict]:
    if idxs.size == 0:
        return []
    prices = df["close"].to_numpy()
    pts = np.array([(i, float(prices[i])) for i in idxs if 0 <= i < len(prices)], dtype=float)
    if pts.size == 0:
        return []
    pts = pts[np.argsort(pts[:,1])]
    clusters = []
    cur_c = pts[0,1]; cur_m = [int(pts[0,0])]
    for i in range(1, len(pts)):
        ii = int(pts[i,0]); px = pts[i,1]
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

# -------------------- 핵심 추정 (한 행) --------------------

def estimate_one(symbol: str, sig_ts: pd.Timestamp, side_hint: Optional[str],
                 timeframe: str, lookback: int, lookahead: int,
                 assume_tz: Optional[str],
                 roots: Optional[List[str]], patterns: Optional[List[str]]) -> Optional[Dict]:
    # 1) OHLCV 로드: sr_engine -> 로컬 CSV
    df = None
    if SE_HAVE and se_get_ohlcv is not None:
        try:
            df = se_get_ohlcv(symbol, timeframe)
        except Exception:
            df = None
    if df is None or len(df) == 0:
        df = _load_ohlcv_local(symbol, timeframe, assume_tz, roots, patterns)
        if df is None or len(df) == 0:
            print(f"[EST][{symbol}] OHLCV not found for tf={timeframe}")
            return None
    try:
        df = ensure_ts_col(df, assume_tz)
    except Exception as ex:
        print(f"[EST][{symbol}] ensure_ts_col failed: {ex}")
        return None

    # 2) 알람 봉 인덱스
    sig_ts = to_utc_ts(sig_ts)
    ts64 = series_to_ns_utc(df["ts"])
    key64 = np.int64(sig_ts.value)
    i = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i < 0:
        print(f"[EST][{symbol}] ts out of range: {sig_ts} < {df['ts'].min()}")
        return None
    sig_price = float(df["close"].iloc[i])

    # 3) 분석 구간
    i0 = max(0, i - lookback)
    i1 = min(len(df)-1, i + lookahead)
    seg = df.iloc[i0:i1+1].reset_index(drop=True)
    if seg.empty or len(seg) < 5:
        print(f"[EST][{symbol}] segment too small around {sig_ts} (len={len(seg)})")
        return None

    # 4) 밴드/스윙/클러스터
    try:
        band = float(_auto_band(seg))
    except Exception:
        band = 1e-9
    try:
        swings_raw = _find_swings(seg)
    except Exception:
        swings_raw = _find_swings_fallback(seg, window=5)
    sw = _normalize_swings(swings_raw, len(seg))
    sup_levels = _cluster_by_price(seg, sw["low_idx"], band)
    res_levels = _cluster_by_price(seg, sw["high_idx"], band)

    cands = []
    if side_hint == "support":
        for lv in sup_levels:
            cands.append(("support", lv))
    elif side_hint == "resistance":
        for lv in res_levels:
            cands.append(("resistance", lv))
    else:
        for lv in sup_levels:
            cands.append(("support", lv))
        for lv in res_levels:
            cands.append(("resistance", lv))
    if not cands:
        print(f"[EST][{symbol}] no clusters near {sig_ts} (side={side_hint})")
        return None

    # 5) 가장 가까운 레벨
    best = None; best_dist = None
    for side, lv in cands:
        center = float(lv["center"])
        dist = abs(sig_price - center) / max(1e-9, sig_price)
        if (best is None) or (dist < best_dist):
            best = (side, lv); best_dist = dist
    side_sel, lv_sel = best
    out = {
        "est_level": float(lv_sel["center"]),
        "est_band": float(band),
        "est_touches": int(lv_sel.get("touches", 0)),
        "distance_pct": float(best_dist * 100.0),
        "method": f"swings_cluster[{lookback}/{lookahead}]",
        "side_used": side_sel,
        "sig_price": float(sig_price),
    }
    return out

# -------------------- MP 러너 --------------------

def worker(task: Dict) -> Dict:
    row = task["row"]
    symbol = task["symbol"]
    timeframe = task["timeframe"]
    ts = task["ts"]
    side_hint = task["side_hint"]
    lookback = task["lookback"]
    lookahead = task["lookahead"]
    assume_tz = task["assume_tz"]
    roots = task["roots"]
    patterns = task["patterns"]

    est = None
    try:
        est = estimate_one(symbol, ts, side_hint, timeframe, lookback, lookahead,
                           assume_tz, roots, patterns)
    except Exception as ex:
        print(f"[EST][{symbol}] worker exception: {ex}")
        est = None

    out = dict(row)
    if est:
        out.update(est)
    else:
        out.update({
            "est_level": np.nan,
            "est_band": np.nan,
            "est_touches": np.nan,
            "distance_pct": np.nan,
            "method": "failed",
            "side_used": side_hint or "",
            "sig_price": np.nan,
        })
    return out

def run_pool_map(func, tasks, procs: int, chunksize: int):
    from multiprocessing import Pool
    with Pool(processes=procs) as pool:
        for res in pool.imap_unordered(func, tasks, chunksize=chunksize):
            yield res

# -------------------- 메인 --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv 경로")
    ap.add_argument("--timeframe", default="15m", help="기본 타임프레임(행에 없을 때)")
    ap.add_argument("--lookback", type=int, default=400, help="알람시점 이전 탐색 봉 수")
    ap.add_argument("--lookahead", type=int, default=40, help="알람시점 이후 탐색 봉 수")
    ap.add_argument("--limit", type=int, default=0, help="상위 N행만 처리(디버그)")
    ap.add_argument("--procs", type=int, default=8, help="프로세스 수")
    ap.add_argument("--chunksize", type=int, default=128, help="imap_unordered chunksize")
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv", help="출력 CSV 경로")
    # 선택: 로컬 CSV 탐색 커스터마이즈
    ap.add_argument("--assume-ohlcv-tz", default=None, help="로컬 OHLCV가 naive일 때 가정할 타임존 (예: UTC, Asia/Seoul)")
    ap.add_argument("--ohlcv-roots", default=None, help="세미콜론(;) 구분 루트 목록")
    ap.add_argument("--ohlcv-patterns", default=None, help="세미콜론(;) 구분 패턴 목록 (예: data/ohlcv/{symbol}-{timeframe}.csv;...)")
    args = ap.parse_args()

    if not os.path.exists(args.signals):
        raise FileNotFoundError(args.signals)

    df = pd.read_csv(args.signals)
    if "ts" not in df.columns:
        raise ValueError("signals 파일에 ts 컬럼 필요(UTC ISO-string 권장).")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "timeframe" not in df.columns:
        df["timeframe"] = args.timeframe
    df["timeframe"] = df["timeframe"].fillna(args.timeframe).astype(str)
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker":"symbol"})
        else:
            raise ValueError("signals 파일에 symbol 컬럼 필요.")

    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    roots = None
    patterns = None
    if args.ohlcv_roots:
        roots = [s.strip() for s in args.ohlcv_roots.split(";") if s.strip()]
    if args.ohlcv_patterns:
        patterns = [s.strip() for s in args.ohlcv_patterns.split(";") if s.strip()]

    rows = []
    for k, row in df.iterrows():
        rows.append({
            "row": dict(row),
            "symbol": str(row["symbol"]).strip(),
            "timeframe": str(row.get("timeframe", args.timeframe) or args.timeframe),
            "ts": row["ts"],
            "side_hint": parse_side(row),
            "lookback": args.lookback,
            "lookahead": args.lookahead,
            "assume_tz": args.assume_ohlcv_tz,
            "roots": roots,
            "patterns": patterns,
        })

    total = len(rows)
    print(f"[EST] start MP: rows={total}, timeframe(default)={args.timeframe}, look={args.lookback}/{args.lookahead}, procs={args.procs}, chunksize={args.chunksize}")

    t0 = time.time()
    out_iter = run_pool_map(worker, rows, procs=args.procs, chunksize=args.chunksize)
    out_list = []
    cnt = 0
    for out in out_iter:
        out_list.append(out)
        cnt += 1
        if cnt % 50 == 0 or cnt == total:
            print(f"[EST] processed {cnt}/{total} ...")
            sys.stdout.flush()

    out_df = pd.DataFrame(out_list)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    ok = int(out_df["method"].ne("failed").sum()) if "method" in out_df.columns else 0
    took = time.time() - t0
    print(f"[EST] done. saved -> {args.out}")
    print(f"[EST] success {ok}/{len(out_df)} rows, took {took:.1f}s")

if __name__ == "__main__":
    main()