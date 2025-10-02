#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone

import requests

LOGS_DIR = "./logs"
PRICES_CSV_DEFAULT = os.path.join(LOGS_DIR, "prices.csv")
UNIVERSE_PATH_DEFAULT = "./configs/universe.txt"
os.makedirs(LOGS_DIR, exist_ok=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def append_prices(path: str, rows: list[dict]):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts","symbol","price","src"])
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def load_universe(symbols: str | None = None, symbols_file: str | None = None) -> list[str]:
    out: list[str] = []
    if symbols:
        for s in symbols.split(","):
            s = s.strip().upper()
            if s:
                out.append(s)
    path = symbols_file or (UNIVERSE_PATH_DEFAULT if not out else None)
    if path:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    out.append(line.upper())
        else:
            logging.warning("universe file not found: %s (feeder will not fetch anything)", path)
    # Upbit 마켓코드: KRW-XXX / BTC-XXX / USDT-XXX ...
    return out

def fetch_upbit_tickers(symbols: list[str]) -> dict[str, float]:
    """Upbit public REST: GET https://api.upbit.com/v1/ticker?markets=KRW-BTC,KRW-NEAR,..."""
    if not symbols:
        return {}
    # Upbit 형식은 대문자 'KRW-BTC' 그대로 사용
    markets = ",".join(symbols)
    url = "https://api.upbit.com/v1/ticker"
    try:
        r = requests.get(url, params={"markets": markets}, timeout=3)
        r.raise_for_status()
        data = r.json()
        out = {}
        for item in data:
            market = item.get("market","")
            trade = item.get("trade_price", None)
            if market and trade is not None:
                out[market.upper()] = float(trade)
        return out
    except Exception as e:
        logging.warning("upbit fetch error: %s", e)
        return {}

def main():
    ap = argparse.ArgumentParser(description="Upbit price feeder (universe-aware)")
    ap.add_argument("--prices-csv", default=PRICES_CSV_DEFAULT)
    ap.add_argument("--symbols", help="comma-separated (e.g. KRW-BTC,KRW-NEAR)")
    ap.add_argument("--symbols-file", help=f"path to universe file (default: {UNIVERSE_PATH_DEFAULT})")
    ap.add_argument("--interval", type=int, default=1, help="seconds between polls (default: 1)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    uni = load_universe(args.symbols, args.symbols_file)
    if not uni:
        logging.warning("universe is EMPTY -> feeder will idle until symbols provided")
    else:
        logging.info("universe loaded (%d): %s", len(uni), ", ".join(uni))

    logging.info("writing to %s (interval=%ds)", args.prices_csv, args.interval)

    while True:
        if uni:
            quotes = fetch_upbit_tickers(uni)
            if quotes:
                ts = now_iso()
                rows = [{"ts": ts, "symbol": sym, "price": f"{px:.12g}", "src": "upbit"} for sym, px in quotes.items()]
                append_prices(args.prices_csv, rows)
                logging.info("tick: %d symbols appended", len(rows))
        time.sleep(max(1, args.interval))

if __name__ == "__main__":
    main()