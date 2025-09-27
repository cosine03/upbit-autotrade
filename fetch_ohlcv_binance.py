# -*- coding: utf-8 -*-
import argparse, os, time, sys
import pandas as pd
import ccxt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True, help="한 줄에 하나씩 심볼명 (예: BTCUSDT)")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--since-days", type=int, default=180, help="과거 N일 전부터")
    ap.add_argument("--outdir", default="./data/ohlcv_binance")
    ap.add_argument("--throttle", type=float, default=0.25)  # 초
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--append", action="store_true", help="기존 CSV가 있으면 최신부터 이어붙이기")
    ap.add_argument("--max-bars", type=int, default=0, help="0이면 무제한, >0이면 상한")
    return ap.parse_args()

def ms():
    import time
    return int(time.time() * 1000)

def load_existing(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df[["ts","open","high","low","close","volume"]].reset_index(drop=True)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} (rows={len(df)})  first={df['ts'].iloc[0]}  last={df['ts'].iloc[-1]}")

def fetch_all(exchange, symbol, timeframe, since_ms, throttle, retries, max_bars=0):
    """안전 루프: 마지막 바 기준으로 since를 진행. 중복/정지 방지."""
    all_rows = []
    total = 0
    last_ts = None
    LIMIT = 1000

    while True:
        for r in range(retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=LIMIT)
                break
            except Exception as e:
                if r == retries - 1:
                    raise
                time.sleep(1.0)
        if not ohlcv:
            # 더 이상 없음
            break

        # 진행(진척) 확인: 마지막 바 기준
        last = ohlcv[-1][0]
        if last_ts is not None and last <= last_ts:
            # 진척 없음 -> 종료
            break
        last_ts = last

        all_rows.extend(ohlcv)
        total += len(ohlcv)
        if max_bars and total >= max_bars:
            break

        # 다음 페이지: 마지막 바 타임스탬프 + 1ms
        since_ms = last + 1
        time.sleep(throttle)

    # pandas로 정리
    if not all_rows:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
    return df

def main():
    args = parse_args()
    with open(args.symbols_file, "r", encoding="utf-8") as f:
        symbols = [s.strip() for s in f if s.strip()]
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()

    # since_ms 계산
    since_ms = ms() - args.since_days * 24 * 60 * 60 * 1000

    for sym in symbols:
        out = os.path.join(args.outdir, f"{sym}-{args.timeframe}.csv")
        base_df = None
        since = since_ms

        if args.append:
            old = load_existing(out)
            if old is not None and not old.empty:
                base_df = old
                # 기존 마지막 ts 이후부터 이어받기
                last_old_ms = int(old["ts"].iloc[-1].value // 10**6)  # ns -> ms
                since = last_old_ms + 1
                print(f"[APPEND] {sym} resume since {pd.to_datetime(since, unit='ms', utc=True)} (old_last={old['ts'].iloc[-1]})")
            else:
                print(f"[APPEND] {sym} no existing file; start fresh.")

        print(f"[FETCH] {sym} tf={args.timeframe} since={pd.to_datetime(since, unit='ms', utc=True)}")
        df = fetch_all(ex, sym, args.timeframe, since, args.throttle, args.retries, args.max_bars)

        if base_df is not None and not base_df.empty:
            if not df.empty:
                df = pd.concat([base_df, df], ignore_index=True)
            else:
                df = base_df

        if df.empty:
            print(f"[WARN] {sym} fetched 0 rows")
            continue

        # 컬럼 타입 안전화
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)

        save_csv(df, out)

if __name__ == "__main__":
    main()