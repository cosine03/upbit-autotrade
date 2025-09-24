# -*- coding: utf-8 -*-
"""
estimate_tv_levels.py
- TradingView(폴 지표) 알람 시점의 레벨(가격대)을 근사 추정해서 signals_tv.csv를 보강합니다.
- 방식: 알람 시점 전후의 OHLCV 구간에서 스윙탐지 → 가격 클러스터링 → 알람 당시 가격과 가장 가까운 레벨 선택
- 출력: ./logs/signals_tv_enriched.csv (기존 컬럼 + est_level, est_band, est_touches, distance_pct, method)

필요:
  - sr_engine.data.get_ohlcv
  - sr_engine.levels.auto_deviation_band, find_swings
  - pandas, numpy, python-dotenv (선택: TIMEFRAME 기본값 .env에서 읽음)

사용 예:
  python estimate_tv_levels.py ./logs/signals_tv.csv --timeframe 15m --lookback 400 --lookahead 40
  (timeframe은 로그에 없다면 전체 기본값으로 사용, 있으면 각 행의 timeframe 우선)

주의:
  - 모든 타임스탬프는 UTC로 맞춥니다.
  - OHLCV가 일시적으로 None일 수 있으므로 재시도하고, 실패 시 해당 행은 건너뜁니다.
"""

import os
import re
import time
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 최소 의존 (이미 프로젝트에 존재)
from sr_engine.data import get_ohlcv
from sr_engine.levels import auto_deviation_band, find_swings

# -------------------- 공통 유틸 --------------------

def to_utc_ts(x) -> pd.Timestamp:
    """문자열/타임스탬프를 UTC-aware Timestamp로 강제 변환."""
    if isinstance(x, pd.Timestamp):
        # pandas Timestamp
        return x.tz_convert("UTC") if x.tz is not None else x.tz_localize("UTC")
    # str/datetime 등
    return pd.to_datetime(x, utc=True, errors="coerce")

def ensure_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """df에 tz-aware 'ts' 컬럼 보장 (OHLCV index 기준)."""
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
    """tz-aware datetime 시리즈를 ns 정수 ndarray로 변환 (FutureWarning 없이)."""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    # ✅ 경고 제거: Series.view(...) 대신 astype("int64")
    return s.astype("int64", copy=False).to_numpy()

def floor_to_frame(ts_utc: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """봉 시작 시간으로 내림."""
    s = timeframe.lower().strip()
    if s.endswith("m"):
        step = int(s[:-1])
    elif s.endswith("h"):
        step = int(s[:-1]) * 60
    elif s.endswith("d"):
        step = int(s[:-1]) * 60 * 24
    else:
        step = 15
    q = (ts_utc.minute // step) * step
    return ts_utc.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(minutes=q)

def parse_side(row: pd.Series) -> Optional[str]:
    """row에서 support/resistance 추출 (열 또는 메시지에서)."""
    side = str(row.get("side", "") or "").strip().lower()
    if side in ("support", "resistance"):
        return side
    msg = str(row.get("message", "") or "")
    if re.search(r"resistance", msg, re.I):
        return "resistance"
    if re.search(r"support", msg, re.I):
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

# -------------------- 핵심 추정 로직 --------------------

def estimate_level_for_signal(
    symbol: str,
    sig_ts: pd.Timestamp,
    side_hint: Optional[str],
    timeframe: str,
    lookback: int = 400,
    lookahead: int = 40,
) -> Optional[Dict]:
    """
    한 개 알람에 대한 레벨 근사 추정.
    - sig_ts: 알람 시각(UTC aware)
    - side_hint: 'support'/'resistance'/None
    반환: {est_level, est_band, est_touches, distance_pct, method, side_used, sig_price}
    """
    # 1) OHLCV 로드 & ts 정규화
    df = get_ohlcv(symbol, timeframe)
    if df is None or len(df) == 0:
        return None
    df = ensure_ts_col(df)
    ts = df["ts"]

    # 2) 알람 시점이 포함되는 봉 찾기
    sig_ts = to_utc_ts(sig_ts)  # UTC aware
    ts64 = series_to_ns_utc(ts)
    key64 = np.int64(sig_ts.value)  # ns
    i = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i < 0:
        return None

    # 알람 직전/해당 봉의 종가를 알람 당시 가격 proxy로 사용
    sig_price = float(df["close"].iloc[i])

    # 3) 분석 구간 슬라이스
    i0 = max(0, i - lookback)
    i1 = min(len(df) - 1, i + lookahead)
    seg = df.iloc[i0:i1 + 1].reset_index(drop=True)
    if seg.empty or len(seg) < 5:
        return None

    # 4) 밴드/스윙/클러스터
    band = float(auto_deviation_band(seg))
    try:
        swings_raw = find_swings(seg)
    except TypeError:
        swings_raw = find_swings(seg, window=5)
    sw = normalize_swings(swings_raw, len(seg))
    sup_levels = cluster_by_price(seg, sw["low_idx"], band)
    res_levels = cluster_by_price(seg, sw["high_idx"], band)

    # 5) 후보군 선택
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
        return None

    # 6) 알람 당시 가격에 가장 가까운 레벨 선택
    best = None
    best_dist = None
    for side, lv in cands:
        center = float(lv["center"])
        dist = abs(sig_price - center) / max(1e-9, sig_price)
        if (best is None) or (dist < best_dist):
            best = (side, lv)
            best_dist = dist

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

# -------------------- 메인 파이프라인 --------------------

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv 경로")
    ap.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "15m"), help="기본 타임프레임(행에 없을 때)")
    ap.add_argument("--lookback", type=int, default=400, help="알람 시점 이전 탐색 봉 수")
    ap.add_argument("--lookahead", type=int, default=40, help="알람 시점 이후 탐색 봉 수")
    ap.add_argument("--limit", type=int, default=0, help="상위 N행만 처리(디버그용)")
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv", help="출력 CSV 경로")
    args = ap.parse_args()

    path = args.signals
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # TV 로그만 (source 컬럼이 있다면)
    if "source" in df.columns:
        df = df[
            df["source"].astype(str).str.upper().str.contains("TV|TRADINGVIEW", na=False)
            | (df["source"].isna())
        ].copy()

    # ts/timeframe 정규화
    if "ts" not in df.columns:
        raise ValueError("signals 파일에 'ts' 컬럼이 필요합니다 (UTC ISO-string 추천).")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "timeframe" not in df.columns:
        df["timeframe"] = args.timeframe
    df["timeframe"] = df["timeframe"].fillna(args.timeframe).astype(str)

    # 심볼 컬럼 이름 통일
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df.rename(columns={"ticker": "symbol"}, inplace=True)
        else:
            raise ValueError("signals 파일에 'symbol' 컬럼이 필요합니다.")

    # 처리 제한
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    rows = []
    t0 = time.time()
    total = len(df)
    print(f"[EST] start: rows={total}, timeframe(default)={args.timeframe}, look={args.lookback}/{args.lookahead}")

    for k, row in df.iterrows():
        sym = str(row["symbol"]).strip()
        tf  = str(row.get("timeframe", args.timeframe) or args.timeframe)
        ts  = row["ts"]
        if pd.isna(ts) or not sym:
            continue

        side = parse_side(row)

        try:
            est = estimate_level_for_signal(
                symbol=sym,
                sig_ts=ts,
                side_hint=side,
                timeframe=tf,
                lookback=args.lookback,
                lookahead=args.lookahead,
            )
        except Exception:
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
                "side_used": side or "",
                "sig_price": np.nan,
            })
        rows.append(out)

        if (k + 1) % 50 == 0:
            print(f"[EST] processed {k+1}/{total} ...")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    took = time.time() - t0
    ok = out_df["method"].ne("failed").sum()
    print(f"[EST] done. saved -> {args.out}")
    print(f"[EST] success {ok}/{len(out_df)} rows, took {took:.1f}s")

if __name__ == "__main__":
    main()