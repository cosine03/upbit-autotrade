#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, os
from collections import defaultdict

# src 우선순위 (값이 클수록 우선)
SRC_RANK = {
    "upbit": 100,
    "ws": 90,
    "csv": 80,
    "rest": 70,
    "ohlcv": 10,
}

def src_rank(s: str) -> int:
    return SRC_RANK.get((s or "").lower(), 0)

def main():
    ap = argparse.ArgumentParser(description="Merge multiple prices.csv into one snapshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    store = {}  # key: (ts, symbol) -> (price, src)
    n_in = 0
    for path in args.inputs:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            assert set(r.fieldnames) >= {"ts","symbol","price","src"}, f"bad header in {path}"
            for row in r:
                ts = row["ts"].strip()
                sym = row["symbol"].strip().upper()
                price = row["price"].strip()
                src = (row.get("src") or "").strip()
                key = (ts, sym)

                # 기존과 우선순위 비교
                if key in store:
                    old_price, old_src = store[key]
                    if src_rank(src) > src_rank(old_src):
                        store[key] = (price, src)
                else:
                    store[key] = (price, src)
                n_in += 1

    # 정렬: ts, symbol
    keys_sorted = sorted(store.keys(), key=lambda x: (x[0], x[1]))
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts","symbol","price","src"])
        w.writeheader()
        for ts, sym in keys_sorted:
            price, src = store[(ts, sym)]
            w.writerow({"ts": ts, "symbol": sym, "price": price, "src": src})

    print(f"merged rows: {len(keys_sorted)} from inputs={len(args.inputs)} -> {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()