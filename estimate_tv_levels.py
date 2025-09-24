# estimate_tv_levels.py (robust)
# -------------------------------------------------------------------
# signals_tv.csv를 읽어 신호 시점 가격/레벨을 추정하고 distance_pct(%)를 추가 저장.
# - 경로 탐색 강화(glob)
# - tz-naive OHLCV용 타임존 가정 옵션(--assume-ohlcv-tz)
# - 실패 사유(est_note) 집계 출력
# -------------------------------------------------------------------

import argparse, os, sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import glob

# ---------- args ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("signals", help="signals_tv.csv")
    p.add_argument("--timeframe", default="15m", help="CSV에 없거나 공백일 때 기본 timeframe")
    p.add_argument("--out", default="./logs/signals_tv_enriched.csv", help="출력 CSV 경로")

    # 네가 넘기던 옵션들(로컬판에서 기능적으로 쓰진 않지만 허용)
    p.add_argument("--http-timeout", type=float, default=8.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--throttle", type=float, default=0.10)

    # *** 중요: OHLCV 타임존 가정 (tz-naive일 때 적용)
    p.add_argument("--assume-ohlcv-tz", default="UTC",
                   help="OHLCV ts가 tz-naive일 때 가정할 타임존 (예: UTC, Asia/Seoul)")

    # 레벨 추정 파라미터
    p.add_argument("--lookback-bars", type=int, default=300)
    p.add_argument("--q-support", type=float, default=0.05)
    p.add_argument("--q-resistance", type=float, default=0.95)
    return p.parse_args()

# ---------- utils ----------
def log(msg): print(msg, flush=True)

# timeframe 정규화/별칭
_TF_ALIASES = {
    "1m": ["1m","1","1min","1t"],
    "3m": ["3m","3","3min","3t"],
    "5m": ["5m","5","5min","5t"],
    "10m": ["10m","10","10min","10t"],
    "15m": ["15m","15","15min","15t"],
    "30m": ["30m","30","30min","30t"],
    "1h": ["1h","60m","60","1hr","1hour"],
    "2h": ["2h","120m","120","2hr","2hours"],
    "4h": ["4h","240m","240","4hr","4hours"],
    "8h": ["8h","480m","480","8hr","8hours"],
    "12h":["12h","720m","720","12hr","12hours"],
    "1d": ["1d","1day","1440m","1440","d","day"],
}
def normalize_tf(tf: str) -> str:
    t = str(tf).strip().lower()
    for k, al in _TF_ALIASES.items():
        if t in al: return k
    return t  # 모르는 값은 그대로

def to_utc_series(x, assume_tz: str="UTC"):
    """
    Series/array/iterable -> UTC Timestamp Series
    tz-aware는 UTC 변환, tz-naive는 assume_tz로 localize 후 UTC 변환.
    """
    s = pd.to_datetime(pd.Series(x), errors="coerce", utc=False)
    # tz-aware와 tz-naive 혼재 가능 -> 개별 처리
    out = []
    for ts in s:
        if pd.isna(ts):
            out.append(pd.NaT)
        else:
            if ts.tzinfo is None:
                try:
                    ts_local = ts.tz_localize(assume_tz)
                except Exception:
                    # 잘못된 tz 문자열일 수 있어, fallback: UTC로 가정
                    ts_local = ts.tz_localize("UTC")
                out.append(ts_local.tz_convert("UTC"))
            else:
                out.append(ts.tz_convert("UTC"))
    return pd.Series(out, dtype="datetime64[ns, UTC]")

def normalize_signals(df: pd.DataFrame, default_tf: str) -> pd.DataFrame:
    need = ["ts","event","side","symbol","timeframe"]
    for c in need:
        if c not in df.columns: df[c] = np.nan
    for c in ["event","side","symbol","timeframe"]:
        df[c] = df[c].astype(str).str.strip()
    df.loc[(df["timeframe"]== "") | (df["timeframe"].str.lower()=="nan"), "timeframe"] = default_tf
    df["timeframe"] = df["timeframe"].map(normalize_tf)

    df["event"]  = df["event"].str.lower()
    df["side"]   = df["side"].str.lower()
    df["symbol"] = df["symbol"].str.replace(r"\s+","", regex=True)

    df["ts"] = to_utc_series(df["ts"], assume_tz="UTC")  # TV 신호는 보통 UTC로 왔다고 가정
    keep = df["ts"].notna() & df["symbol"].ne("") & df["timeframe"].ne("") & df["event"].ne("") & df["side"].ne("")
    return df.loc[keep].reset_index(drop=True)

# ---------- OHLCV loading ----------
def candidate_patterns(symbol: str, tf: str) -> List[str]:
    # 다양한 파일명/폴더 패턴 지원
    # ex) KRW-ADA_15m.csv, KRW-ADA-15m.csv, upbit_KRW-ADA_15m.csv, 폴더분할 등
    bases = [
        "./ohlcv", "./ohlcv_tv", "./data/ohlcv", "./logs/ohlcv", "./data", "./logs"
    ]
    seps = ["_", "-"]
    tf_alts = _TF_ALIASES.get(tf, []) + [tf]
    names = []
    for b in bases:
        for alt in tf_alts:
            for sep in seps:
                names.append(f"{b}/{symbol}{sep}{alt}.csv")
                names.append(f"{b}/upbit_{symbol}{sep}{alt}.csv")
                names.append(f"{b}/upbit/{symbol}{sep}{alt}.csv")
                names.append(f"{b}/{tf}/{symbol}.csv")
                names.append(f"{b}/{symbol}/{alt}.csv")
    # glob 하위까지 스캔: 파일명에 symbol/tf가 모두 포함되는 후보
    globs = []
    for root in bases:
        globs += glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)

    # symbol, tf 키워드 둘 다 들어간 것만 추가
    sym_key = symbol.lower()
    tf_keys = set([x.lower() for x in tf_alts])
    for path in globs:
        low = path.lower().replace("\\","/")
        if sym_key in low and any(tk in low for tk in tf_keys):
            names.append(path)
    # 중복 제거, 존재하는 것만
    uniq = []
    seen = set()
    for p in names:
        if p in seen: continue
        seen.add(p)
        if os.path.isfile(p):
            uniq.append(p)
    return uniq

def load_ohlcv(symbol: str, timeframe: str, assume_tz: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    cands = candidate_patterns(symbol, timeframe)
    if not cands:
        return None, f"not_found:{symbol}_{timeframe}"
    # 첫 일치 파일 사용
    path = cands[0]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"read_csv:{e}"

    for c in ["ts","open","high","low","close"]:
        if c not in df.columns:
            return None, f"missing_col:{c}"
    # tz 보정
    df["ts"] = to_utc_series(df["ts"], assume_tz=assume_tz)
    df = df.dropna(subset=["ts","open","high","low","close"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        return None, "empty_after_clean"

    return df, None

# ---------- core ----------
def search_index_at_or_before(ts_series: pd.Series, key_ts: pd.Timestamp) -> int:
    a = pd.to_datetime(ts_series, utc=True).astype("datetime64[ns]").to_numpy()
    k = pd.to_datetime(key_ts, utc=True)
    if pd.isna(k): return -1
    k64 = np.datetime64(k.to_datetime64())
    idx = int(np.searchsorted(a, k64, side="right")) - 1
    return idx

def estimate_level(window_df: pd.DataFrame, side: str, q_s: float, q_r: float) -> Tuple[float, str]:
    if window_df.empty:
        return np.nan, "empty_window"
    if side == "support":
        return float(np.nanquantile(window_df["low"].to_numpy(), q_s)), ""
    if side == "resistance":
        return float(np.nanquantile(window_df["high"].to_numpy(), q_r)), ""
    return np.nan, f"unknown_side:{side}"

# ---------- main ----------
def main():
    args = parse_args()

    try:
        df_raw = pd.read_csv(args.signals)
    except Exception as e:
        log(f"[EST] ERROR read signals: {e}")
        sys.exit(1)

    df = normalize_signals(df_raw.copy(), args.timeframe)
    valid_events = {"level2_detected","level3_detected","price_in_box","box_breakout","line_breakout"}
    df = df[df["event"].isin(valid_events)].reset_index(drop=True)

    # 결과 컬럼
    df["price_at_signal"] = np.nan
    df["est_level"] = np.nan
    df["distance_pct"] = np.nan
    df["bars_lookback_used"] = np.nan
    df["est_note"] = ""

    if df.empty:
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        log(f"[EST] done. saved -> {args.out}")
        log(f"[EST] success 0/0 rows")
        return

    # 캐시
    cache = {}
    ok = 0
    total = len(df)
    df = df.sort_values(["symbol","timeframe","ts"]).reset_index(drop=True)

    for i, r in df.iterrows():
        sym = r["symbol"]
        tf  = normalize_tf(r["timeframe"])
        ts  = r["ts"]
        side= r["side"]

        key = (sym, tf)
        if key not in cache:
            oh, err = load_ohlcv(sym, tf, assume_tz=args.assume_ohlcv_tz)
            cache[key] = (oh, err)
        else:
            oh, err = cache[key]

        if oh is None:
            df.at[i,"est_note"] = f"ohlcv:{err}"
            continue

        idx = search_index_at_or_before(oh["ts"], ts)
        if idx < 0:
            df.at[i,"est_note"] = "index_not_found_before_ts"
            continue

        price = float(oh.at[idx,"close"])
        df.at[i,"price_at_signal"] = price

        i0 = max(0, idx - int(args.lookback_bars) + 1)
        win = oh.iloc[i0:idx+1]
        level, note = estimate_level(win, side, args.q_support, args.q_resistance)
        if not np.isfinite(level) or level<=0:
            df.at[i,"est_note"] = note if note else "invalid_level"
            continue

        df.at[i,"est_level"] = level
        df.at[i,"bars_lookback_used"] = len(win)
        df.at[i,"distance_pct"] = abs(price - level) / level * 100.0
        ok += 1

    # 저장
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log(f"[EST] done. saved -> {args.out}")
    log(f"[EST] success {ok}/{total} rows")

    # 원인 요약
    vc = df["est_note"].fillna("").value_counts()
    if len(vc):
        log("[EST] notes summary (top 10):")
        for k, v in vc.head(10).items():
            kshow = k if k else "(empty)"
            log(f"  - {kshow}: {v}")

    # 문제된 심볼/TF 몇 개 힌트 출력
    if ok < total:
        bad = df[df["est_note"].ne("")]
        if not bad.empty:
            ex = bad.groupby(["symbol","timeframe"])["est_note"].head(1).reset_index(drop=True)
            log("[EST] examples of issues:")
            for row in ex.head(10):
                pass
            # 더 간단히: 심볼/TF별 첫 에러 표시
            for (sym, tf), sub in bad.groupby(["symbol","timeframe"]):
                log(f"  - {sym} {tf}: {sub['est_note'].iloc[0]}")
                # 10개만
                if len(ex.head(10))==10: break

if __name__ == "__main__":
    main()