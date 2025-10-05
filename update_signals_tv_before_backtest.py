#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_signals_tv_before_backtest.py

Prepare signals_tv_full.csv for backtesting:
- Normalize columns
- Clean message text
- Filter by symbol prefix / timeframe
- Optionally exclude InternalHost or localhost
- Deduplicate by key (ts|symbol|event|side|level|touches)
- Save as signals_tv_ready.csv (UTF-8)

Example:
python update_signals_tv_before_backtest.py \
  --input  "D:\upbit_autotrade_starter\logs\signals_tv_full.csv" \
  --output "D:\upbit_autotrade_starter\logs\signals_tv_ready.csv" \
  --symbol-prefix "KRW-" --timeframe "15m" \
  --exclude-internalhost --include-localhost
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

REQUIRED_COLS = [
    "ts", "event", "side", "level", "touches", "symbol",
    "timeframe", "extra", "host", "message", "distance_pct"
]

INTERNAL_HOST_PATTERN = re.compile(r"System\.Management\.Automation\.Internal\.Host", re.I)

def clean_message(m: str) -> str:
    if not m:
        return ""
    m = re.sub(r"\s*\|\s*[_?]+\s*", " | ", m)
    m = re.sub(r"^\s*[_?]+\s*", "", m)
    m = re.sub(r"\?{2,}", "?", m)
    m = re.sub(r"\s{2,}", " ", m.strip())
    return m

def canon_row(r: dict) -> dict:
    side = (r.get("side") or r.get("type") or "").strip()
    msg_raw = (r.get("message") or r.get("details") or r.get("source") or "")
    msg_clean = clean_message(msg_raw)
    host_v = (r.get("host") or r.get("origin") or r.get("source") or "").strip()
    if INTERNAL_HOST_PATTERN.search(host_v):
        host_v = "InternalHost"

    tf = (r.get("timeframe") or "").strip()
    extra = (r.get("extra") or "").strip()
    dist = (r.get("distance_pct") or "").strip()

    out = {
        "ts": (r.get("ts") or "").strip(),
        "event": (r.get("event") or "").strip(),
        "side": side,
        "level": (r.get("level") or "").strip(),
        "touches": (r.get("touches") or "").strip(),
        "symbol": (r.get("symbol") or "").strip(),
        "timeframe": tf,
        "extra": extra,
        "host": host_v,
        "message": msg_clean,
        "distance_pct": dist,
    }
    for c in REQUIRED_COLS:
        out.setdefault(c, "")
    return out

def key_of(rec: dict, keep_host_in_key: bool = False) -> str:
    base = "{ts}|{symbol}|{event}|{side}|{level}|{touches}".format(**{k: rec.get(k, "") for k in [
        "ts", "symbol", "event", "side", "level", "touches"
    ]})
    if keep_host_in_key:
        return base + "|" + (rec.get("host") or "")
    return base

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=r"D:\upbit_autotrade_starter\logs\signals_tv_full.csv")
    ap.add_argument("--output", default=r"D:\upbit_autotrade_starter\logs\signals_tv_ready.csv")
    ap.add_argument("--symbol-prefix", default="", help="Filter by symbol prefix, e.g., KRW-")
    ap.add_argument("--timeframe", default="", help="Filter by timeframe, e.g., 15m (empty = no filter)")
    ap.add_argument("--exclude-internalhost", action="store_true", help="Exclude InternalHost rows")
    ap.add_argument("--include-localhost", action="store_true", help="Include 127.0.0.1 rows (default: exclude)")
    ap.add_argument("--keep-host-in-key", action="store_true", help="Include host in dedup key")
    return ap.parse_args()

def main():
    args = parse_args()
    src = Path(args.input)
    if not src.exists():
        print(f"[ERR] input not found: {src}", file=sys.stderr)
        sys.exit(1)

    with src.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    canon = [canon_row(r) for r in rows]

    if args.symbol_prefix:
        canon = [r for r in canon if r["symbol"].startswith(args.symbol_prefix)]

    if args.timeframe:
        canon = [r for r in canon if (not r["timeframe"]) or (r["timeframe"] == args.timeframe)]

    if args.exclude_internalhost:
        canon = [r for r in canon if r["host"] != "InternalHost"]

    if not args.include_localhost:
        canon = [r for r in canon if r["host"] != "127.0.0.1"]

    seen = set()
    dedup = []
    for r in canon:
        k = key_of(r, keep_host_in_key=args.keep_host_in_key)
        if k not in seen:
            seen.add(k)
            dedup.append(r)

    def ts_key(rec):
        t = rec.get("ts", "")
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            return t
    dedup.sort(key=ts_key)

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in dedup:
            writer.writerow({c: r.get(c, "") for c in REQUIRED_COLS})

    print(f"[OK] saved: {dst} (rows={len(dedup)})")
    by_event = {}
    for r in dedup:
        by_event[r["event"]] = by_event.get(r["event"], 0) + 1
    top = sorted(by_event.items(), key=lambda x: x[1], reverse=True)
    print("[SUMMARY] events:", ", ".join(f"{k}:{v}" for k, v in top[:6]))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERR] {e}", file=sys.stderr)
        sys.exit(1)