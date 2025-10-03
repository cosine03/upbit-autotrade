#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, os, re
from datetime import datetime, timezone

def parse_ts(s):
    # Upbit: '2025-09-28T12:34:00' (UTC) 또는 '2025-09-28 12:34:00'
    ss = s.strip().replace(" ", "T")
    if ss.endswith("Z"):
        dt = datetime.fromisoformat(ss.replace("Z","+00:00"))
    else:
        # 대부분 csv는 TZ 없는 UTC로 저장됐을 가능성 → UTC 가정
        dt = datetime.fromisoformat(ss).replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()

def detect_schema(headers):
    h = [x.lower() for x in headers]
    # Upbit 공식 필드 예: candle_date_time_utc, opening_price, high_price, low_price, trade_price, candle_acc_trade_price, ...
    if "candle_date_time_utc" in h and "trade_price" in h:
        return "upbit_api"
    # 일반 OHLCV: timestamp/date, open, high, low, close, volume
    if ("timestamp" in h or "date" in h) and "close" in h:
        return "generic_ohlcv"
    # 일부 툴: time, open, high, low, close, volume
    if "time" in h and "close" in h:
        return "generic_ohlcv2"
    return "unknown"

def row_to_price(schema, row, headers):
    hl = {k.lower():k for k in headers}
    if schema == "upbit_api":
        ts = row[hl["candle_date_time_utc"]]
        px = row[hl["trade_price"]]
    elif schema in ("generic_ohlcv","generic_ohlcv2"):
        key_ts = "timestamp" if "timestamp" in hl else ("date" if "date" in hl else "time")
        ts = row[hl[key_ts]]
        px = row[hl["close"]]
    else:
        return None
    try:
        ts_iso = parse_ts(ts)
        price = float(px)
        return ts_iso, price
    except Exception:
        return None

def infer_symbol_from_path(path, default_symbol=None):
    # 파일명에서 KRW-XXX 등을 추정
    m = re.search(r'(KRW|BTC|USDT)-[A-Z0-9]+', os.path.basename(path).upper())
    if m:
        return m.group(0)
    return default_symbol or "KRW-BTC"

def main():
    ap = argparse.ArgumentParser(description="Convert OHLCV CSVs to prices snapshot")
    ap.add_argument("--out", required=True, help="output prices csv path")
    ap.add_argument("--symbol", help="force symbol (if cannot infer from filename)")
    ap.add_argument("inputs", nargs="+", help="one or more OHLCV csv files")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wrote_header = False
    n_rows = 0

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=["ts","symbol","price","src"])
        w.writeheader(); wrote_header = True
        for path in args.inputs:
            if not os.path.exists(path):
                continue
            symbol = infer_symbol_from_path(path, args.symbol)
            with open(path, "r", newline="", encoding="utf-8") as f:
                r0 = csv.reader(f)
                headers = next(r0, None)
                if not headers: continue
                schema = detect_schema(headers)
                if schema == "unknown":
                    continue
                r = csv.DictReader(open(path, "r", newline="", encoding="utf-8"))
                for row in r:
                    conv = row_to_price(schema, row, headers)
                    if not conv: continue
                    ts_iso, price = conv
                    w.writerow({"ts": ts_iso, "symbol": symbol, "price": f"{price:.12g}", "src": "ohlcv"})
                    n_rows += 1

    print(f"written {n_rows} rows -> {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()