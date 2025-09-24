# -*- coding: utf-8 -*-
"""
estimate_tv_levels_mp.py
- TradingView(폴 지표) 알람 시점의 레벨(가격대)을 근사 추정해서 signals_tv.csv를 보강 (멀티프로세싱).
- 방식: 알람 시점 전후 OHLCV 구간에서 스윙탐지 → 가격 클러스터링 → 알람 당시 가격과 가장 가까운 레벨 선택
- 출력: 지정 경로 CSV (기존 컬럼 + est_level, est_band, est_touches, distance_pct, method, side_used, sig_price)

사용 예:
  python estimate_tv_levels_mp.py .\logs\signals_tv.csv ^
    --timeframe 15m ^
    --lookback 400 --lookahead 40 ^
    --procs 12 --chunksize 128 ^
    --ohlcv-roots ".;.\data;.\data\ohlcv;.\ohlcv;.\logs;.\logs\ohlcv" ^
    --ohlcv-patterns "data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;ohlcv/{symbol}-{timeframe}.csv;ohlcv/{symbol}_{timeframe}.csv;logs/ohlcv/{symbol}-{timeframe}.csv;logs/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv" ^
    --assume-ohlcv-tz UTC ^
    --out .\logs\signals_tv_enriched.csv

메모:
- 모든 타임스탬프는 UTC로 맞춤.
- OHLCV CSV 컬럼명은 대소문자/표기 약간 달라도 자동 인식 시도 (ts/open/high/low/close/volume).
- distance_pct는 %로 저장 (예: 0.83).
"""

import os
import re
import time
import argparse
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# -------------------- 공통 유틸 --------------------

def to_utc_ts(x) -> pd.Timestamp:
    """문자열/타임스탬프를 UTC-aware Timestamp로 강제 변환."""
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def datetime_series_to_ns(s: pd.Series) -> np.ndarray:
    """
    tz-aware datetime 시리즈를 ns 정수 ndarray로 변환.
    astype 경고 없이 안전하게 처리.
    """
    s = pd.to_datetime(s, utc=True, errors="coerce")
    # epoch 차이로 int64(ns) 변환
    base = pd.Timestamp("1970-01-01", tz="UTC")
    return (s - base).view("int64")

def ensure_ts_col(df: pd.DataFrame, assume_tz: Optional[str] = None) -> pd.DataFrame:
    """df에 tz-aware 'ts' 컬럼 보장 (UTC)."""
    out = df.copy()
    if "ts" in out.columns:
        ts = pd.to_datetime(out["ts"], errors="coerce", utc=True)
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("OHLCV must have DatetimeIndex or 'ts' column")
        ts = out.index
        if ts.tz is None:
            # 로컬 tz 가정 후 UTC로 변환
            if assume_tz:
                ts = ts.tz_localize(assume_tz).tz_convert("UTC")
            else:
                ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    out["ts"] = ts
    return out.reset_index(drop=True)

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

# -------------------- OHLCV 로더 --------------------

def map_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    다양한 CSV 헤더를 ts/open/high/low/close/volume로 표준화.
    """
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    ts_col = pick("ts", "timestamp", "time", "date")
    o_col = pick("open", "o")
    h_col = pick("high", "h")
    l_col = pick("low", "l")
    c_col = pick("close", "c", "price")
    v_col = pick("volume", "vol")

    need = [ts_col, o_col, h_col, l_col, c_col]
    if any(x is None for x in need):
        raise ValueError("OHLCV columns not recognized")

    out = pd.DataFrame({
        "ts": df[ts_col],
        "open": pd.to_numeric(df[o_col], errors="coerce"),
        "high": pd.to_numeric(df[h_col], errors="coerce"),
        "low": pd.to_numeric(df[l_col], errors="coerce"),
        "close": pd.to_numeric(df[c_col], errors="coerce"),
    })
    if v_col:
        out["volume"] = pd.to_numeric(df[v_col], errors="coerce")
    return out

def load_ohlcv_local(symbol: str,
                     timeframe: str,
                     roots: List[str],
                     patterns: List[str],
                     assume_tz: Optional[str]) -> Optional[pd.DataFrame]:
    """
    패턴 목록/루트 목록을 돌며 첫번째로 존재하는 파일을 읽어 표준화된 OHLCV로 반환.
    """
    sym = symbol.strip()
    tf  = timeframe.strip()
    cands = []
    for pat in patterns:
        rel = pat.format(symbol=sym, timeframe=tf)
        # pat 에 디렉토리가 포함되어 있을 수 있음 → 각 root와 결합
        for root in roots:
            root = root.strip().strip('"').strip("'")
            path = os.path.normpath(os.path.join(root, rel))
            cands.append(path)

    for path in cands:
        if os.path.exists(path):
            try:
                df_raw = pd.read_csv(path)
                df = map_ohlcv_columns(df_raw)
                df = ensure_ts_col(df, assume_tz)
                # 결측/정렬/중복 제거
                df = df.dropna(subset=["ts","open","high","low","close"]).sort_values("ts")
                df = df.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
                return df
            except Exception:
                continue
    return None

# -------------------- 레벨 추정(스윙/클러스터) --------------------

def auto_deviation_band(seg: pd.DataFrame) -> float:
    """
    변동성 기준 밴드 (robust): median absolute deviation 기반.
    """
    px = seg["close"].to_numpy(dtype=float)
    med = np.median(px)
    mad = np.median(np.abs(px - med)) if len(px) else 0.0
    # 1.4826 * MAD ~ 표준편차 근사
    st = 1.4826 * mad if mad > 0 else np.std(px) if len(px) else 0.0
    # 과도한 0 방지 + 스케일
    band = max(1e-9, st * 0.75)
    return float(band)

def find_swings(seg: pd.DataFrame, window: int = 3) -> Dict[str, np.ndarray]:
    """
    간단한 스윙 탐지: window 내 국소 극값(저점/고점) 인덱스 반환.
    """
    n = len(seg)
    if n < 2*window + 1:
        return {"low_idx": np.array([], dtype=int), "high_idx": np.array([], dtype=int)}
    lo = []
    hi = []
    low = seg["low"].to_numpy(dtype=float)
    high = seg["high"].to_numpy(dtype=float)
    for i in range(window, n - window):
        if low[i] == np.min(low[i-window:i+window+1]):
            lo.append(i)
        if high[i] == np.max(high[i-window:i+window+1]):
            hi.append(i)
    return {"low_idx": np.array(lo, dtype=int), "high_idx": np.array(hi, dtype=int)}

def cluster_by_price(seg: pd.DataFrame, idxs: np.ndarray, band: float) -> List[Dict]:
    """
    가격 근접도(band)로 간단 클러스터링.
    """
    if idxs.size == 0:
        return []
    prices = seg["close"].to_numpy(dtype=float)
    pts = np.array([(i, float(prices[i])) for i in idxs if 0 <= i < len(prices)], dtype=float)
    if pts.size == 0:
        return []
    pts = pts[np.argsort(pts[:, 1])]

    clusters = []
    cur_c = pts[0, 1]
    cur_m = [int(pts[0, 0])]
    for k in range(1, len(pts)):
        ii = int(pts[k, 0]); px = pts[k, 1]
        if abs(px - cur_c) <= band:
            cur_m.append(ii)
            cur_c = (cur_c * (len(cur_m)-1) + px) / len(cur_m)
        else:
            clusters.append({"center": float(cur_c), "idx": np.array(sorted(set(cur_m)), dtype=int)})
            cur_c = px
            cur_m = [ii]
    clusters.append({"center": float(cur_c), "idx": np.array(sorted(set(cur_m)), dtype=int)})
    for c in clusters:
        c["touches"] = int(len(c["idx"]))
    return clusters

def estimate_level_for_signal(df_ohlcv: pd.DataFrame,
                              sig_ts: pd.Timestamp,
                              side_hint: Optional[str],
                              timeframe: str,
                              lookback: int,
                              lookahead: int) -> Optional[Dict]:
    """
    한 개 알람에 대한 레벨 근사 추정.
    반환: {est_level, est_band, est_touches, distance_pct(%), method, side_used, sig_price}
    """
    # 1) OHLCV 준비
    df = df_ohlcv
    ts = df["ts"]
    sig_ts = to_utc_ts(sig_ts)

    # 2) 알람 포함 봉 찾기 (알람 직전/동일 봉 close 사용)
    ts64 = datetime_series_to_ns(ts)
    key64 = np.int64(sig_ts.value)  # ns
    i = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i < 0:
        return None
    sig_price = float(df["close"].iloc[i])

    # 3) 분석 구간
    i0 = max(0, i - lookback)
    i1 = min(len(df)-1, i + lookahead)
    seg = df.iloc[i0:i1+1].reset_index(drop=True)
    if seg.empty or len(seg) < 5:
        return None

    # 4) 스윙/클러스터
    band = float(auto_deviation_band(seg))
    sw = find_swings(seg, window=3)
    sup_levels = cluster_by_price(seg, sw["low_idx"], band)
    res_levels = cluster_by_price(seg, sw["high_idx"], band)

    # 5) 후보군
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

    # 6) 가장 가까운 레벨 선택
    best, best_dist = None, None
    for side, lv in cands:
        center = float(lv["center"])
        dist = abs(sig_price - center) / max(1e-9, sig_price)
        if (best is None) or (dist < best_dist):
            best = (side, lv)
            best_dist = dist

    side_sel, lv_sel = best
    return {
        "est_level": float(lv_sel["center"]),
        "est_band": float(band),
        "est_touches": int(lv_sel.get("touches", 0)),
        "distance_pct": float(best_dist * 100.0),  # 퍼센트
        "method": f"swings_cluster[{lookback}/{lookahead}]",
        "side_used": side_sel,
        "sig_price": float(sig_price),
    }

# -------------------- 멀티프로세싱 워커 --------------------

def _worker_symbol(args):
    (symbol, df_rows, timeframe, lookback, lookahead,
     roots, patterns, assume_tz) = args

    # 심볼 OHLCV 로딩 (한 번)
    ohlcv = load_ohlcv_local(symbol, timeframe, roots, patterns, assume_tz)
    if ohlcv is None or ohlcv.empty:
        # 해당 심볼 전부 실패 처리
        out_rows = []
        for _, row in df_rows.iterrows():
            out = dict(row)
            out.update({
                "est_level": np.nan, "est_band": np.nan, "est_touches": np.nan,
                "distance_pct": np.nan, "method": "failed(no_ohlcv)",
                "side_used": parse_side(row) or "", "sig_price": np.nan,
            })
            out_rows.append(out)
        return pd.DataFrame(out_rows)

    out_rows = []
    for _, row in df_rows.iterrows():
        ts = to_utc_ts(row["ts"])
        side = parse_side(row)
        try:
            est = estimate_level_for_signal(
                df_ohlcv=ohlcv,
                sig_ts=ts,
                side_hint=side,
                timeframe=timeframe,
                lookback=lookback,
                lookahead=lookahead,
            )
        except Exception:
            est = None

        r = dict(row)
        if est:
            r.update(est)
        else:
            r.update({
                "est_level": np.nan, "est_band": np.nan, "est_touches": np.nan,
                "distance_pct": np.nan, "method": "failed", "side_used": side or "",
                "sig_price": np.nan,
            })
        out_rows.append(r)

    return pd.DataFrame(out_rows)

# -------------------- 메인 --------------------

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv 경로")
    ap.add_argument("--timeframe", default="15m", help="기본 타임프레임(행에 없을 때)")
    ap.add_argument("--lookback", type=int, default=400, help="알람 시점 이전 탐색 봉 수")
    ap.add_argument("--lookahead", type=int, default=40, help="알람 시점 이후 탐색 봉 수")
    ap.add_argument("--limit", type=int, default=0, help="상위 N행만 처리(디버그용)")

    ap.add_argument("--procs", type=int, default=max(1, cpu_count() // 2), help="프로세스 병렬 수")
    ap.add_argument("--chunksize", type=int, default=128, help="심볼 작업 청크 사이즈(내부)")

    ap.add_argument("--ohlcv-roots", default=".;.\\data;.\\data\\ohlcv;.\\ohlcv;.\\logs;.\\logs\\ohlcv",
                    help="세미콜론(;) 구분 루트 목록")
    ap.add_argument("--ohlcv-patterns", default="data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;"
                                                "ohlcv/{symbol}-{timeframe}.csv;ohlcv/{symbol}_{timeframe}.csv;"
                                                "logs/ohlcv/{symbol}-{timeframe}.csv;logs/ohlcv/{symbol}_{timeframe}.csv;"
                                                "{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv",
                    help="세미콜론(;) 구분 패턴 목록")
    ap.add_argument("--assume-ohlcv-tz", default="UTC", help="OHLCV 파일의 타임존 가정(없으면 이 TZ로 로컬라이즈 후 UTC로 변환)")

    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv", help="출력 CSV 경로")

    args = ap.parse_args()

    path = args.signals
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError("signals 파일에 'ts' 컬럼이 필요합니다.")

    # TV 로그만 필터링(있으면)
    if "source" in df.columns:
        m = df["source"].astype(str).str.upper().str.contains("TV|TRADINGVIEW", na=False)
        df = df[m | df["source"].isna()].copy()

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "timeframe" not in df.columns:
        df["timeframe"] = args.timeframe
    df["timeframe"] = df["timeframe"].fillna(args.timeframe).astype(str)

    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        else:
            raise ValueError("signals 파일에 'symbol' 컬럼이 필요합니다.")

    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    # NaT / 심볼 결측 제거
    df = df.dropna(subset=["ts", "symbol"]).reset_index(drop=True)

    roots = [s.strip().strip('"').strip("'") for s in str(args.ohlcv_roots).split(";") if s.strip()]
    patterns = [s.strip().strip('"').strip("'") for s in str(args.ohlcv_patterns).split(";") if s.strip()]

    # 심볼 단위로 그룹핑 → 각 워커는 심볼 OHLCV 한 번만 로드
    groups = []
    for sym, g in df.groupby("symbol", sort=False):
        # 해당 심볼의 timeframe은 행마다 있을 수 있지만, 대부분 동일하므로 대표값 사용
        tf = str(g["timeframe"].iloc[0] if "timeframe" in g.columns else args.timeframe)
        groups.append((sym, g.reset_index(drop=True), tf, args.lookback, args.lookahead,
                       roots, patterns, args.assume_ohlcv_tz))

    t0 = time.time()
    print(f"[EST] start MP: rows={len(df)}, timeframe(default)={args.timeframe}, look={args.lookback}/{args.lookahead}, "
          f"procs={args.procs}, chunksize={args.chunksize}")

    out_parts = []
    ok_rows = 0
    fail_noohlcv = 0
    with Pool(processes=max(1, int(args.procs))) as pool:
        for part in pool.imap_unordered(_worker_symbol, groups, chunksize=max(1, int(args.chunksize))):
            if part is None or part.empty:
                continue
            out_parts.append(part)
            ok_rows += part["method"].ne("failed").sum()
            fail_noohlcv += (part["method"] == "failed(no_ohlcv)").sum()
            # 간단 진행 로그
            if len(out_parts) % 5 == 0:
                done = sum(len(p) for p in out_parts)
                print(f"[EST] processed {done}/{len(df)} ... OK so far: {ok_rows}")

    out_df = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=list(df.columns) + [
        "est_level","est_band","est_touches","distance_pct","method","side_used","sig_price"
    ])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    took = time.time() - t0
    total = len(out_df)
    ok = out_df["method"].ne("failed").sum() if total else 0
    print(f"[EST] done. saved -> {args.out}")
    print(f"[EST] success {ok}/{total} rows, took {took:.1f}s (no_ohlcv={fail_noohlcv})")

if __name__ == "__main__":
    main()