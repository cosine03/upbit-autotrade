# -*- coding: utf-8 -*-
"""
fetch_ohlcv_upbit.py (paginated)
- Upbit OHLCV를 since~now 까지 페이징으로 반복 요청해서 충분한 캔들을 확보합니다.
- 기존 파일이 있을 경우 병합(중복 제거, 시간 정렬) 후 저장합니다.

사용 예:
  python fetch_ohlcv_upbit.py --symbols-file .\symbols.txt --timeframe 15m --since-days 180 \
    --max-bars 20000 --outdir .\data\ohlcv

출력 파일명:
  {OUTDIR}/{SYMBOL}-{TIMEFRAME}.csv  (예: .\data\ohlcv\KRW-BTC-15m.csv)

CSV 컬럼:
  ts, open, high, low, close, volume  (ts는 UTC ISO8601)
"""

import os
import time
import argparse
from typing import List, Optional
import math
import ccxt
import pandas as pd

# ------------ helpers ------------

TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "10m": 600_000, "15m": 900_000,
    "30m": 1_800_000, "60m": 3_600_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "1w": 604_800_000
}

def tf_to_ms(tf: str) -> int:
    s = tf.strip().lower()
    if s in TF_MS:
        return TF_MS[s]
    if s.endswith("m"):
        return int(s[:-1]) * 60_000
    if s.endswith("h"):
        return int(s[:-1]) * 3_600_000
    if s.endswith("d"):
        return int(s[:-1]) * 86_400_000
    raise ValueError(f"unsupported timeframe: {tf}")

def save_csv(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_csv(path, index=False)

def load_existing(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def merge_ohlcv(old_df: Optional[pd.DataFrame], new_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ts","open","high","low","close","volume"]
    if old_df is not None and set(cols).issubset(old_df.columns):
        merged = pd.concat([old_df[cols], new_df[cols]], ignore_index=True)
    else:
        merged = new_df[cols].copy()

    # 중복 제거 + 시간 정렬
    merged["ts"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["ts"]).drop_duplicates(subset=["ts"]).sort_values("ts")
    # ISO8601로 저장
    merged["ts"] = merged["ts"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return merged.reset_index(drop=True)

# ------------ core fetch ------------

def fetch_symbol_ohlcv_paginated(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_days: int,
    max_bars: int = 20000,
    batch: int = 200,
    sleep_sec: float = 0.08,
    retries: int = 3,
) -> pd.DataFrame:
    tfms = tf_to_ms(timeframe)
    now_ms = ex.milliseconds()
    since_ms = now_ms - since_days * 86_400_000

    out = []
    cursor = since_ms
    total = 0

    while True:
        if total >= max_bars:
            break

        got = None
        for _ in range(retries):
            try:
                got = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=batch)
                break
            except ccxt.RateLimitExceeded:
                time.sleep(max(0.2, sleep_sec * 2))
            except Exception:
                time.sleep(sleep_sec)

        if not got:
            break

        # ccxt 표준: [timestamp, open, high, low, close, volume]
        for t, o, h, l, c, v in got:
            out.append([t, o, h, l, c, v])
        total += len(got)

        # 다음 페이지로 전진
        last_ts = got[-1][0]
        next_cursor = last_ts + tfms
        if next_cursor <= cursor:
            # 안전장치: 타임스탬프가 안 전진하면 중단
            break
        cursor = next_cursor

        # 이제(now)보다 충분히 가까우면 종료
        # (여유 2캔들)
        if cursor >= now_ms - 2 * tfms:
            break

        time.sleep(sleep_sec)

    if not out:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    df = pd.DataFrame(out, columns=["ts_ms","open","high","low","close","volume"])
    # ts를 UTC ISO로
    ts = pd.to_datetime(df["ts_ms"], utc=True, unit="ms")
    df = pd.DataFrame({
        "ts": ts.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
    })
    return df

# ------------ CLI ------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True, help="라인별 심볼 목록 (예: KRW-BTC)")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--since-days", type=int, default=90)
    ap.add_argument("--max-bars", type=int, default=20000, help="심볼당 최대 캔들 수 상한")
    ap.add_argument("--batch", type=int, default=200, help="요청당 최대 캔들 (Upbit 일반적으로 200)")
    ap.add_argument("--sleep", type=float, default=0.08, help="요청 사이 대기(초)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--outdir", default="./data/ohlcv")
    args = ap.parse_args()

    with open(args.symbols_file, "r", encoding="utf-8") as f:
        symbols = [ln.strip() for ln in f if ln.strip()]

    ex = ccxt.upbit({"enableRateLimit": True})
    print(f"[FETCH] start: symbols={len(symbols)}, timeframe={args.timeframe}, since_days={args.since_days}, outdir={args.outdir}")

    for sym in symbols:
        path = os.path.join(args.outdir, f"{sym}-{args.timeframe}.csv")
        try:
            new_df = fetch_symbol_ohlcv_paginated(
                ex, sym, args.timeframe,
                since_days=args.since_days,
                max_bars=args.max_bars,
                batch=args.batch,
                sleep_sec=args.sleep,
                retries=args.retries,
            )
            old_df = load_existing(path)
            merged = merge_ohlcv(old_df, new_df)
            save_csv(path, merged)
            print(f"[{sym}] saved -> {path} (rows={len(merged)})")
        except Exception as e:
            print(f"[{sym}] ERROR: {e}")

    print("[FETCH] done.")

if __name__ == "__main__":
    main()