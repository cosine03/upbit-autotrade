# -*- coding: utf-8 -*-
"""
Fetch OHLCV from Binance via ccxt
Example:
  python fetch_ohlcv_binance.py --symbols-file ./configs/binance_universe.txt \
    --timeframe 15m --since-days 90 --max-bars 3000 \
    --throttle 0.6 --retries 3 --outdir ./data/ohlcv_binance
"""

import argparse, os, time, sys
import pandas as pd
import ccxt

def fetch_symbol(exchange, symbol, timeframe, since_ms, limit, retries, throttle):
    for attempt in range(retries):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df
        except Exception as e:
            print(f"[WARN] fetch fail {symbol} (try {attempt+1}/{retries}): {e}")
            time.sleep(throttle)
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True, help="text file with one symbol per line (e.g. BTC/USDT)")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--since-days", type=int, default=90)
    ap.add_argument("--max-bars", type=int, default=3000)
    ap.add_argument("--throttle", type=float, default=0.6)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--outdir", default="./data/ohlcv_binance")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    exchange = ccxt.binance({"enableRateLimit": True})

    since_ms = exchange.milliseconds() - args.since_days * 24 * 60 * 60 * 1000

    with open(args.symbols_file) as f:
        symbols = [s.strip() for s in f.readlines() if s.strip()]

    print(f"[FETCH] start: symbols={len(symbols)}, timeframe={args.timeframe}, since_days={args.since_days}, outdir={args.outdir}")

    for sym in symbols:
        # ccxt Binance 표준 심볼 포맷은 "BTC/USDT"
        sym_ccxt = sym if "/" in sym else sym.replace("USDT", "/USDT")
        path = os.path.join(args.outdir, f"{sym.replace('/','')}-{args.timeframe}.csv")

        df = fetch_symbol(exchange, sym_ccxt, args.timeframe, since_ms, args.max_bars, args.retries, args.throttle)
        if df.empty:
            print(f"[SKIP] {sym_ccxt} (no data)")
            continue

        df.to_csv(path, index=False)
        print(f"[OK] {sym} -> {path} (rows={len(df)})")
        time.sleep(args.throttle)

    print("[DONE] all symbols")

if __name__ == "__main__":
    main()