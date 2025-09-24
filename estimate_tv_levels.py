# -*- coding: utf-8 -*-
"""
estimate_tv_levels.py (Single Process, Full Patched, backward-compatible)
- TradingView 알람 시점의 레벨을 근사 추정해서 signals_tv.csv를 보강
- sr_engine.data.get_ohlcv 구버전/신버전 모두 호환:
  * 신버전: get_ohlcv(symbol, timeframe, roots=..., patterns=..., assume_tz=...)
  * 구버전: get_ohlcv(symbol, timeframe) + 로컬 CSV 로더(roots/patterns) 폴백
"""

import os
import re
import time
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from sr_engine.levels import auto_deviation_band, find_swings
from sr_engine.data import get_ohlcv as _core_get_ohlcv  # 이름 충돌 방지

# -------------------- 로컬 CSV 로더 --------------------
def _load_local_ohlcv(symbol: str, timeframe: str,
                      roots: List[str], patterns: List[str],
                      assume_tz: str = "UTC") -> Optional[pd.DataFrame]:
    """roots+patterns 조합으로 CSV를 찾아 읽어 DatetimeIndex(+tz)로 반환."""
    for root in roots:
        for pat in patterns:
            rel = pat.format(symbol=symbol, timeframe=timeframe)
            path = os.path.join(root, rel)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    # ts 컬럼/인덱스 정규화
                    if "ts" in df.columns:
                        ts = pd.to_datetime(df["ts"], utc=False, errors="coerce")
                        # 파일이 assume_tz 기준의 ‘로컬시간’일 수 있으므로 먼저 tz_localize
                        try:
                            ts = ts.dt.tz_localize(assume_tz, nonexistent="shift_forward", ambiguous="NaT")
                        except Exception:
                            # 이미 tz-aware면 넘어감
                            if ts.dt.tz is None:
                                ts = ts.dt.tz_localize("UTC")
                        ts = ts.dt.tz_convert("UTC")
                        df.index = ts
                    else:
                        # 흔한 컬럼명 케이스 대응
                        if "time" in df.columns:
                            ts = pd.to_datetime(df["time"], utc=False, errors="coerce")
                        elif "datetime" in df.columns:
                            ts = pd.to_datetime(df["datetime"], utc=False, errors="coerce")
                        else:
                            # 첫 컬럼이 시간일 가능성
                            first_col = df.columns[0]
                            ts = pd.to_datetime(df[first_col], utc=False, errors="coerce")
                        try:
                            ts = ts.dt.tz_localize(assume_tz, nonexistent="shift_forward", ambiguous="NaT")
                        except Exception:
                            if ts.dt.tz is None:
                                ts = ts.dt.tz_localize("UTC")
                        ts = ts.dt.tz_convert("UTC")
                        df.index = ts

                    # 칼럼 표준화
                    cols_map = {
                        "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
                        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
                    }
                    lower_map = {c: c.lower() for c in df.columns}
                    df.rename(columns={c: cols_map.get(lower_map[c], lower_map[c]) for c in df.columns}, inplace=True)
                    need = {"open", "high", "low", "close"}
                    if not need.issubset(set(df.columns)):
                        continue

                    # 정렬/결측 제거
                    df = df.sort_index()
                    df = df[["open", "high", "low", "close"] + ([c for c in ["volume"] if c in df.columns])]
                    df = df.dropna(subset=["open", "high", "low", "close"])
                    return df
                except Exception:
                    continue
    return None

def get_ohlcv_compat(symbol: str, timeframe: str,
                     roots: List[str], patterns: List[str],
                     assume_tz: str = "UTC") -> Optional[pd.DataFrame]:
    """신/구버전 get_ohlcv 모두 지원 + 로컬 CSV 폴백."""
    # 1) 신버전 시도
    try:
        df = _core_get_ohlcv(symbol, timeframe, roots=roots, patterns=patterns, assume_tz=assume_tz)
        if df is not None and len(df) > 0:
            return df
    except TypeError:
        # 신버전 시그니처가 아님 → 구버전일 수 있음
        pass
    except Exception:
        pass

    # 2) 구버전 시도
    try:
        df = _core_get_ohlcv(symbol, timeframe)
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass

    # 3) 로컬 CSV 폴백
    return _load_local_ohlcv(symbol, timeframe, roots, patterns, assume_tz)

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
    return s.astype("int64").to_numpy()  # pandas 2.2: view 대신 astype

def parse_side(row: pd.Series) -> Optional[str]:
    side = str(row.get("side", "") or "").lower()
    if side in ("support", "resistance"):
        return side
    msg = str(row.get("message", "") or "")
    if re.search(r"resistance", msg, re.I):
        return "resistance"
    if re.search(r"support", msg, re.I):
        return "support"
    return None

def cluster_by_price(df: pd.DataFrame, idxs: np.ndarray, band: float) -> List[Dict]:
    if idxs.size == 0:
        return []
    prices = df["close"].to_numpy()
    pts = np.array([(i, float(prices[i])) for i in idxs if 0 <= i < len(prices)], dtype=float)
    if pts.size == 0:
        return []
    pts = pts[np.argsort(pts[:, 1])]

    clusters = []
    cur_c = pts[0, 1]; cur_m = [int(pts[0, 0])]
    for i in range(1, len(pts)):
        ii, px = int(pts[i, 0]), pts[i, 1]
        if abs(px - cur_c) <= band:
            cur_m.append(ii)
            cur_c = (cur_c * (len(cur_m) - 1) + px) / len(cur_m)
        else:
            clusters.append({"center": float(cur_c), "idx": np.array(sorted(set(cur_m)), dtype=int)})
            cur_c, cur_m = px, [ii]
    clusters.append({"center": float(cur_c), "idx": np.array(sorted(set(cur_m)), dtype=int)})
    for c in clusters:
        c["touches"] = int(len(c["idx"]))
    return clusters

def normalize_swings(swings, n_rows: int) -> Dict[str, np.ndarray]:
    def to_int_idx(x):
        if x is None:
            return np.array([], dtype=int)
        arr = np.array(x)
        if arr.dtype.kind not in ("i", "u"):
            arr = pd.to_numeric(arr, errors="coerce").to_numpy()
        arr = arr[np.isfinite(arr)].astype(int, copy=False)
        if n_rows:
            arr = np.clip(arr, 0, n_rows - 1)
        return np.unique(arr)

    if isinstance(swings, dict):
        return {"low_idx": to_int_idx(swings.get("low_idx")),
                "high_idx": to_int_idx(swings.get("high_idx"))}
    if isinstance(swings, (list, tuple)) and len(swings) >= 2:
        return {"low_idx": to_int_idx(swings[0]), "high_idx": to_int_idx(swings[1])}
    return {"low_idx": np.array([], dtype=int), "high_idx": np.array([], dtype=int)}

# -------------------- 핵심 추정 --------------------
def estimate_level_for_signal(symbol: str, sig_ts: pd.Timestamp, side_hint: Optional[str],
                              timeframe: str, lookback: int = 400, lookahead: int = 40,
                              roots=None, patterns=None, assume_tz="UTC") -> Optional[Dict]:
    df = get_ohlcv_compat(symbol, timeframe, roots=roots or [], patterns=patterns or [], assume_tz=assume_tz)
    if df is None or len(df) == 0:
        print(f"[EST][{symbol}] OHLCV not found")
        return None
    df = ensure_ts_col(df)
    ts = df["ts"]

    sig_ts = to_utc_ts(sig_ts)
    ts64 = series_to_ns_utc(ts)
    key64 = np.int64(sig_ts.value)
    i = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i < 0 or i >= len(df):
        print(f"[EST][{symbol}] ts out of range: {sig_ts}")
        return None

    sig_price = float(df["close"].iloc[i])
    i0, i1 = max(0, i - lookback), min(len(df) - 1, i + lookahead)
    seg = df.iloc[i0:i1 + 1].reset_index(drop=True)
    if seg.empty or len(seg) < 5:
        return None

    band = float(auto_deviation_band(seg))
    try:
        swings_raw = find_swings(seg)
    except TypeError:
        swings_raw = find_swings(seg, window=5)
    sw = normalize_swings(swings_raw, len(seg))
    sup_levels = cluster_by_price(seg, sw["low_idx"], band)
    res_levels = cluster_by_price(seg, sw["high_idx"], band)

    cands = []
    if side_hint == "support":
        cands += [("support", lv) for lv in sup_levels]
    elif side_hint == "resistance":
        cands += [("resistance", lv) for lv in res_levels]
    else:
        cands += [("support", lv) for lv in sup_levels] + [("resistance", lv) for lv in res_levels]

    if not cands:
        return None

    best = None; best_dist = None
    for side, lv in cands:
        center = float(lv["center"])
        dist = abs(sig_price - center) / max(1e-9, sig_price)
        if (best is None) or (dist < best_dist):
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

# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv 경로")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--lookahead", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv")
    ap.add_argument("--assume-ohlcv-tz", default="UTC", help="로컬 OHLCV CSV 시간대 (예: UTC, Asia/Seoul)")
    ap.add_argument("--ohlcv-roots", default=".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv")
    ap.add_argument("--ohlcv-patterns", default="{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.signals)
    if "ts" not in df.columns:
        raise ValueError("signals 파일에 'ts' 컬럼 필요")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "timeframe" not in df.columns:
        df["timeframe"] = args.timeframe
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df.rename(columns={"ticker": "symbol"}, inplace=True)
        else:
            raise ValueError("signals 파일에 'symbol' 컬럼 필요")

    if args.limit > 0:
        df = df.iloc[:args.limit].copy()

    roots = [r for r in args.ohlcv_roots.split(";") if r]
    patterns = [p for p in args.ohlcv_patterns.split(";") if p]

    rows = []
    total = len(df)
    t0 = time.time()
    print(f"[EST] start: rows={total}, timeframe={args.timeframe}, look={args.lookback}/{args.lookahead}")

    for k, row in df.iterrows():
        est = estimate_level_for_signal(
            symbol=str(row["symbol"]),
            sig_ts=row["ts"],
            side_hint=parse_side(row),
            timeframe=str(row.get("timeframe", args.timeframe)),
            lookback=args.lookback,
            lookahead=args.lookahead,
            roots=roots,
            patterns=patterns,
            assume_tz=args.assume_ohlcv_tz,
        )
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
                "side_used": "",
                "sig_price": np.nan,
            })
        rows.append(out)
        if (k + 1) % 50 == 0:
            print(f"[EST] processed {k+1}/{total} ...")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    ok = int(out_df["method"].ne("failed").sum())
    print(f"[EST] done. saved -> {args.out}")
    print(f"[EST] success {ok}/{len(out_df)} rows, took {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()