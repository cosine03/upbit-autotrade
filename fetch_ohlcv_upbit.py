# -*- coding: utf-8 -*-
"""
fetch_ohlcv_upbit.py
- Upbit에서 심볼별 OHLCV 캔들(기본: 15m)을 받아 ./data/ohlcv/{symbol}-{timeframe}.csv 로 저장/갱신합니다.
- 기존 파일이 있으면 병합(중복 제거) 후 최신 상태로 유지합니다.

필요: pip install ccxt
예)
  python fetch_ohlcv_upbit.py --symbols KRW-BTC,KRW-ETH,KRW-XRP \
      --timeframe 15m --since-days 30 --limit 2000 --outdir ./data/ohlcv

  python fetch_ohlcv_upbit.py --symbols-file symbols.txt --timeframe 15m
"""

import os
import sys
import time
import argparse
from typing import List, Optional

import ccxt
import pandas as pd

# -------- utils --------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_utc_ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def load_existing(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # 표준 컬럼 보장
        if "ts" in df.columns:
            df["ts"] = to_utc_ts(df["ts"])
        else:
            # 과거 포맷 대응: index가 ts였거나 날짜컬럼이 있을 수 있음
            for cand in ("timestamp","date","time"):
                if cand in df.columns:
                    df["ts"] = to_utc_ts(df[cand])
                    break
            if "ts" not in df.columns:
                return None
        keep = ["ts","open","high","low","close","volume"]
        for k in keep:
            if k not in df.columns:
                df[k] = pd.NA
        df = df[keep].dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        return df
    except Exception:
        return None

def unify_frame_name(tf: str) -> str:
    return tf.strip().lower()

def timeframe_to_ccxt(tf: str) -> str:
    # ccxt 표준 그대로 사용 (m/h/d)
    return unify_frame_name(tf)

def fetch_ohlcv(exchange, symbol: str, timeframe: str, since_ms: Optional[int], limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    # ccxt: [ms, open, high, low, close, volume]
    if not ohlcv:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])
    df = pd.DataFrame(ohlcv, columns=["ms","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    df = df[["ts","open","high","low","close","volume"]]
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

def merge_dedup(old_df: Optional[pd.DataFrame], new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        base = new_df.copy()
    else:
        base = pd.concat([old_df, new_df], ignore_index=True)
    base = base.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    # 타입 정리
    for col in ["open","high","low","close","volume"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    return base.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)

def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    # ISO8601 UTC로 저장
    out = df.copy()
    out["ts"] = out["ts"].dt.tz_convert("UTC")
    out.to_csv(path, index=False)

# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="콤마구분 심볼들 (예: KRW-BTC,KRW-ETH)")
    ap.add_argument("--symbols-file", help="심볼 리스트 파일(한 줄에 하나)")
    ap.add_argument("--timeframe", default="15m", help="기본 15m")
    ap.add_argument("--since-days", type=int, default=45, help="과거 N일 전부터 가져오기 (기본 45일)")
    ap.add_argument("--limit", type=int, default=2000, help="ccxt 요청당 캔들 수 (기본 2000)")
    ap.add_argument("--outdir", default="./data/ohlcv", help="출력 디렉토리")
    ap.add_argument("--sleep", type=float, default=0.2, help="요청 사이 딜레이(초)")

    args = ap.parse_args()

    # 심볼 수집
    symbols: List[str] = []
    if args.symbols:
        symbols += [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file and os.path.exists(args.symbols_file):
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    symbols.append(s)
    symbols = sorted(set(symbols))
    if not symbols:
        print("[FETCH] no symbols provided. use --symbols or --symbols-file")
        sys.exit(2)

    tf = timeframe_to_ccxt(args.timeframe)
    ensure_dir(args.outdir)

    ex = ccxt.upbit()
    # Upbit는 무인증 퍼블릭 OHLCV OK

    # since 계산
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - int(args.since_days * 24 * 60 * 60 * 1000)

    print(f"[FETCH] start: symbols={len(symbols)}, timeframe={tf}, since_days={args.since_days}, outdir={args.outdir}")

    for sym in symbols:
        out_path = os.path.join(args.outdir, f"{sym}-{tf}.csv")
        try:
            old = load_existing(out_path)
            new = fetch_ohlcv(ex, sym, tf, since_ms, args.limit)
            if new is None or new.empty:
                print(f"[{sym}] empty fetch.")
                continue
            merged = merge_dedup(old, new)
            save_csv(merged, out_path)
            print(f"[{sym}] saved -> {out_path} (rows={len(merged)})")
        except Exception as e:
            print(f"[{sym}] ERROR: {e}")
        time.sleep(args.sleep)

    print("[FETCH] done.")

if __name__ == "__main__":
    main()