# -*- coding: utf-8 -*-
"""
Upbit OHLCV fetcher with pagination & incremental merge.

사용 예 (PowerShell):
  python fetch_ohlcv_upbit.py `
    --symbols-file .\symbols.txt `
    --timeframe 15m `
    --since-days 180 `
    --max-bars 20000 `
    --throttle 0.2 `
    --retries 3 `
    --outdir .\data\ohlcv
"""

import os
import time
import math
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    import ccxt
except ImportError:
    raise SystemExit("ccxt가 필요합니다: pip install ccxt")

TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "10m": 600_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

COLS = ["ts", "open", "high", "low", "close", "volume"]

def load_symbols(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        syms = [ln.strip() for ln in f if ln.strip()]
    return syms

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_existing_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # 열 보정
        if "timestamp" in df.columns and "ts" not in df.columns:
            df.rename(columns={"timestamp": "ts"}, inplace=True)
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[COLS].copy()
        # 타입/정렬 보정
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        return df
    except Exception:
        return None

def fetch_paged(exchange, symbol: str, timeframe: str, since_ms: int, max_bars: int,
                throttle: float, retries: int) -> list[list]:
    bars: list[list] = []
    tf_ms = TF_MS[timeframe]
    # ccxt 업빗은 보통 200개 제한
    limit_per_call = 200

    while len(bars) < max_bars:
        last_try_exc = None
        for _ in range(max(1, retries)):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit_per_call)
                break
            except Exception as e:
                last_try_exc = e
                time.sleep(max(0.2, throttle))
        else:
            # retries 모두 실패
            if last_try_exc:
                print(f"[{symbol}] fetch failed after retries: {last_try_exc}")
            break

        if not batch:
            # 더 받을 것이 없음
            break

        # 중복 캔들 제거
        if bars and batch and batch[0][0] == bars[-1][0]:
            batch = batch[1:]

        bars.extend(batch)

        # 다음 페이지
        since_ms = batch[-1][0] + tf_ms

        # 레이트리밋 보호
        time.sleep(throttle)

    # 필요 시 뒤에서 max_bars만 유지
    if len(bars) > max_bars:
        bars = bars[-max_bars:]

    return bars

def make_df(bars: list[list]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[COLS].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True, help="심볼 리스트 텍스트 파일 (한 줄에 하나)")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--since-days", type=int, default=90, help="현재 시각 기준 과거 N일")
    ap.add_argument("--max-bars", type=int, default=2000, help="심볼당 최대 바 수")
    ap.add_argument("--throttle", type=float, default=0.2, help="호출 간 대기(초)")
    ap.add_argument("--retries", type=int, default=3, help="실패 시 재시도 횟수")
    ap.add_argument("--outdir", default="./data/ohlcv")
    args = ap.parse_args()

    symbols = load_symbols(args.symbols_file)
    ensure_dir(args.outdir)

    # ccxt exchange
    exchange = ccxt.upbit({"enableRateLimit": True})

    # since 계산
    since_dt = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    since_ms = int(since_dt.timestamp() * 1000)

    print(f"[FETCH] start: symbols={len(symbols)}, timeframe={args.timeframe}, since_days={args.since_days}, outdir={args.outdir}")

    tf_ms = TF_MS.get(args.timeframe)
    if tf_ms is None:
        raise SystemExit(f"지원하지 않는 timeframe: {args.timeframe}")

    for sym in symbols:
        out_path = os.path.join(args.outdir, f"{sym}-{args.timeframe}.csv")
        old_df = read_existing_csv(out_path)

        # 증분 시작점: 기존 마지막 ts 이후부터
        start_ms = since_ms
        if old_df is not None and not old_df.empty:
            last_ts = int(old_df["ts"].iloc[-1].value // 1_000_000)  # ns→ms
            start_ms = max(start_ms, last_ts + tf_ms)

        bars = fetch_paged(exchange, sym, args.timeframe, start_ms, args.max_bars, args.throttle, args.retries)
        new_df = make_df(bars)

        if old_df is None or old_df.empty:
            merged = new_df
        else:
            if new_df.empty:
                merged = old_df
            else:
                # 경고 회피: 빈 DF concat 피하고, 필요한 열만 보장
                merged = pd.concat([old_df[COLS], new_df[COLS]], ignore_index=True)
                # ts 기준 중복 제거 및 정렬
                merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        # 저장
        ensure_dir(os.path.dirname(out_path))
        merged.to_csv(out_path, index=False)
        print(f"[{sym}] saved -> {out_path} (rows={len(merged)})")

    print("[FETCH] done.")

if __name__ == "__main__":
    main()