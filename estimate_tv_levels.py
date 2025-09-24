# -*- coding: utf-8 -*-
"""
estimate_tv_levels.py  (SP, offline-capable, longest-file pick)
- TradingView 알람 시점의 레벨(가격대)을 근사 추정해서 signals_tv.csv를 보강
- 로컬 OHLCV CSV를 우선 사용(여러 파일이 있으면 "가장 긴" 파일 선택)
- 필요 시 온라인 폴백(sr_engine.data.get_ohlcv)도 가능하지만 --offline 로 차단 가능
- pandas 2.2 경고 제거(Series.view -> astype)
"""

import os
import re
import glob
import time
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# 외부 엔진
from sr_engine.levels import auto_deviation_band, find_swings
from sr_engine.data import get_ohlcv as core_get_ohlcv  # 폴백용

VERSION = "ESTIMATOR-SP v1.2 (offline+longest)"

# -------------------- 유틸 --------------------
def to_utc_ts(x) -> pd.Timestamp:
    """입력을 UTC-aware Timestamp로 변환."""
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def ensure_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """df에 tz-aware 'ts' 컬럼 보장."""
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
    """tz-aware datetime 시리즈를 ns 정수 ndarray로 변환 (pandas 2.2 호환)."""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.astype("int64").to_numpy()

def parse_side(row: pd.Series) -> Optional[str]:
    """row에서 support/resistance 추출 (열 또는 메시지에서)."""
    side = str(row.get("side", "") or "").strip().lower()
    if side in ("support", "resistance"):
        return side
    msg = str(row.get("message", "") or "") + " " + str(row.get("host", "") or "")
    if re.search(r"\bresistance\b", msg, re.I):
        return "resistance"
    if re.search(r"\bsupport\b", msg, re.I):
        return "support"
    return None

def cluster_by_price(df: pd.DataFrame, idxs: np.ndarray, band: float) -> List[Dict]:
    """가격 근접도(band)로 간단 클러스터링."""
    if idxs.size == 0:
        return []
    prices = df["close"].to_numpy()
    pts = np.array([(i, float(prices[i])) for i in idxs if 0 <= i < len(prices)], dtype=float)
    if pts.size == 0:
        return []
    pts = pts[np.argsort(pts[:, 1])]

    clusters = []
    cur_c = pts[0, 1]
    cur_m = [int(pts[0, 0])]
    for i in range(1, len(pts)):
        ii = int(pts[i, 0]); px = pts[i, 1]
        if abs(px - cur_c) <= band:
            cur_m.append(ii)
            cur_c = (cur_c * (len(cur_m) - 1) + px) / len(cur_m)
        else:
            clusters.append({"center": float(cur_c), "idx": np.array(sorted(set(cur_m)), dtype=int)})
            cur_c = px
            cur_m = [ii]
    clusters.append({"center": float(cur_c), "idx": np.array(sorted(set(cur_m)), dtype=int)})
    for c in clusters:
        c["touches"] = int(len(c["idx"]))
    return clusters

def normalize_swings(swings, n_rows: int) -> Dict[str, np.ndarray]:
    """find_swings 반환을 표준 dict로 정규화."""
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
        return {
            "low_idx": to_int_idx(swings.get("low_idx")),
            "high_idx": to_int_idx(swings.get("high_idx")),
        }
    if isinstance(swings, (list, tuple)) and len(swings) >= 2:
        return {"low_idx": to_int_idx(swings[0]), "high_idx": to_int_idx(swings[1])}
    return {"low_idx": np.array([], dtype=int), "high_idx": np.array([], dtype=int)}

# -------------------- 로컬 OHLCV 로딩 --------------------
def _localize_to_utc(series: pd.Series, assume_tz: str) -> pd.Series:
    """문자열/naive datetime을 assume_tz로 localize 후 UTC로 변환. 이미 tz가 있으면 UTC로 변환."""
    s = pd.to_datetime(series, errors="coerce")
    # tz-aware?
    try:
        tz = s.dt.tz
    except Exception:
        tz = None
    if tz is None:
        # naive -> assume_tz -> UTC
        try:
            s = s.dt.tz_localize(assume_tz)
        except Exception:
            # fallback: UTC로 로컬라이즈
            s = s.dt.tz_localize("UTC")
    s = s.dt.tz_convert("UTC")
    return s

def _read_longest_local(symbol: str, timeframe: str, roots: List[str],
                        patterns: List[str], assume_tz: str) -> Optional[pd.DataFrame]:
    """여러 후보 중 '가장 긴' 로컬 파일을 선택해 읽는다."""
    best_df, best_rows, picked = None, -1, None
    for root in roots:
        root = root.strip()
        if not root:
            continue
        for pat in patterns:
            pat = pat.strip()
            if not pat:
                continue
            fn = pat.format(symbol=symbol, timeframe=timeframe)
            path = os.path.join(root, fn)
            for f in glob.glob(path):
                try:
                    df = pd.read_csv(f)
                    # 기대 컬럼: ts, open, high, low, close, volume
                    if "ts" not in df.columns:
                        continue
                    df["ts"] = _localize_to_utc(df["ts"], assume_tz)
                    # 숫자 컬럼 강제
                    for c in ("open", "high", "low", "close", "volume"):
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.dropna(subset=["ts"]).reset_index(drop=True)
                    n = len(df)
                    if n > best_rows:
                        best_rows, best_df, picked = n, df, f
                except Exception:
                    pass
    if best_df is not None:
        print(f"[EST][LOCAL] {symbol} {timeframe} -> {picked} (rows={best_rows})")
    return best_df

def get_ohlcv_compat(symbol: str, timeframe: str,
                     roots: List[str], patterns: List[str],
                     assume_tz: str = "UTC", offline: bool = False) -> Optional[pd.DataFrame]:
    """로컬 최우선, 필요 시 온라인 폴백(offline면 차단)."""
    # 1) 로컬에서 가장 긴 파일 선택
    df = _read_longest_local(symbol, timeframe, roots, patterns, assume_tz)
    if df is not None and not df.empty:
        return df
    # 2) 오프라인 모드면 여기서 종료
    if offline:
        print(f"[EST][{symbol}] no local OHLCV; offline mode -> skip")
        return None
    # 3) 온라인 폴백
    try:
        df = core_get_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            print(f"[EST][{symbol}] online fallback returned empty")
            return None
        return ensure_ts_col(df)
    except Exception as e:
        print(f"[EST][{symbol}] online fallback failed: {e}")
        return None

# -------------------- 핵심 추정 --------------------
def estimate_level_for_signal(symbol: str, sig_ts: pd.Timestamp, side_hint: Optional[str],
                              timeframe: str, lookback: int = 400, lookahead: int = 40,
                              roots=None, patterns=None, assume_tz="UTC", offline=False) -> Optional[Dict]:
    df = get_ohlcv_compat(symbol, timeframe,
                          roots=roots or [], patterns=patterns or [],
                          assume_tz=assume_tz, offline=offline)
    if df is None or len(df) == 0:
        print(f"[EST][{symbol}] OHLCV not found or empty")
        return None

    # ts 정규화 보강
    if "ts" not in df.columns:
        print(f"[EST][{symbol}] missing ts column")
        return None
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)

    if "close" not in df.columns:
        print(f"[EST][{symbol}] missing close column")
        return None

    ts = df["ts"]

    # 알람 시점 인덱스
    sig_ts = to_utc_ts(sig_ts)
    ts64 = series_to_ns_utc(ts)
    key64 = np.int64(sig_ts.value)  # ns
    i = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i < 0 or i >= len(df):
        print(f"[EST][{symbol}] ts out of range: {sig_ts}")
        return None

    sig_price = float(df["close"].iloc[i])

    # 분석 구간
    i0 = max(0, i - lookback)
    i1 = min(len(df) - 1, i + lookahead)
    seg = df.iloc[i0:i1 + 1].reset_index(drop=True)
    if seg.empty or len(seg) < 5:
        return None

    # 밴드/스윙/클러스터
    band = float(auto_deviation_band(seg))
    try:
        swings_raw = find_swings(seg)
    except TypeError:
        swings_raw = find_swings(seg, window=5)
    sw = normalize_swings(swings_raw, len(seg))
    sup_levels = cluster_by_price(seg, sw["low_idx"], band)
    res_levels = cluster_by_price(seg, sw["high_idx"], band)

    # 후보군
    cands = []
    if side_hint == "support":
        cands += [("support", lv) for lv in sup_levels]
    elif side_hint == "resistance":
        cands += [("resistance", lv) for lv in res_levels]
    else:
        cands += [("support", lv) for lv in sup_levels]
        cands += [("resistance", lv) for lv in res_levels]

    if not cands:
        return None

    # 알람 당시 가격에 가장 가까운 레벨
    best, best_dist = None, None
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
    print(f"[{VERSION}]")
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv 경로")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--lookahead", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv")

    # 로컬 OHLCV 검색 경로/패턴 + 오프라인 옵션
    ap.add_argument("--assume-ohlcv-tz", default="UTC", help="로컬 OHLCV CSV 시간대 (예: UTC, Asia/Seoul)")
    ap.add_argument("--ohlcv-roots", default=".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv")
    ap.add_argument("--ohlcv-patterns", default=(
        "data/ohlcv/{symbol}-{timeframe}.csv;"
        "data/ohlcv/{symbol}_{timeframe}.csv;"
        "ohlcv/{symbol}-{timeframe}.csv;"
        "ohlcv/{symbol}_{timeframe}.csv;"
        "logs/ohlcv/{symbol}-{timeframe}.csv;"
        "logs/ohlcv/{symbol}_{timeframe}.csv;"
        "{symbol}-{timeframe}.csv;"
        "{symbol}_{timeframe}.csv"
    ))
    ap.add_argument("--offline", action="store_true", help="로컬 CSV만 사용(네트워크 폴백 금지)")

    args = ap.parse_args()

    # 입력 로드/정규화
    if not os.path.exists(args.signals):
        raise FileNotFoundError(args.signals)
    df = pd.read_csv(args.signals)

    if "ts" not in df.columns:
        raise ValueError("signals 파일에 'ts' 컬럼 필요")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    if "timeframe" not in df.columns:
        df["timeframe"] = args.timeframe
    df["timeframe"] = df["timeframe"].fillna(args.timeframe).astype(str)

    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df.rename(columns={"ticker": "symbol"}, inplace=True)
        else:
            raise ValueError("signals 파일에 'symbol' 컬럼 필요")

    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    roots = [r.strip() for r in str(args.ohlcv_roots).split(";") if r.strip()]
    patterns = [p.strip() for p in str(args.ohlcv_patterns).split(";") if p.strip()]

    total = len(df)
    print(f"[EST] start: rows={total}, timeframe={args.timeframe}, look={args.lookback}/{args.lookahead}")

    rows = []
    t0 = time.time()
    for k, row in df.iterrows():
        est = estimate_level_for_signal(
            symbol=str(row["symbol"]).strip(),
            sig_ts=row["ts"],
            side_hint=parse_side(row),
            timeframe=str(row.get("timeframe", args.timeframe) or args.timeframe),
            lookback=args.lookback,
            lookahead=args.lookahead,
            roots=roots,
            patterns=patterns,
            assume_tz=args.assume_ohlcv_tz,
            offline=bool(args.offline),
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
    ok = out_df["method"].ne("failed").sum()
    took = time.time() - t0
    print(f"[EST] done. saved -> {args.out}")
    print(f"[EST] success {ok}/{len(out_df)} rows, took {took:.1f}s")

if __name__ == "__main__":
    main()