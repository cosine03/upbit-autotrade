#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, csv, glob, re, sys
from collections import defaultdict

# 지원하는 파일명 패턴들:
#  - KRW_ADA_15m.csv  -> KRW-ADA
#  - KRW-ADA_15m.csv  -> KRW-ADA
#  - KRW-ADA-15m.csv  -> KRW-ADA
#  - KRW_ADA.csv      -> KRW-ADA
#  - (필요시 확장 가능)
SYMBOL_PATTERNS = [
    re.compile(r'(?i)\b(KRW)[-_]?([A-Z0-9]+)\b'),  # KRW_ADA, KRW-ADA
]

def infer_symbol_from_path(path: str) -> str | None:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    for pat in SYMBOL_PATTERNS:
        m = pat.search(stem)
        if m:
            return f"{m.group(1).upper()}-{m.group(2).upper()}"
    return None

def locate_cols(header: list[str]) -> dict:
    """
    다양한 헤더 케이스 지원:
      - 시간: ts | timestamp | time
      - 종가: close | c | price
    """
    hmap = {h.strip().lower(): i for i, h in enumerate(header)}

    def find(*cands):
        for c in cands:
            if c in hmap: return hmap[c]
        return None

    idx_ts = find("ts", "timestamp", "time", "datetime")
    idx_close = find("close", "c", "price")

    return {"ts": idx_ts, "close": idx_close}

def read_one_file(path: str) -> tuple[list[tuple[str,str,str]], dict]:
    """
    반환: rows, stats
      rows: List[(ts_iso, symbol, close)]
      stats: {"read": int, "kept": int, "skipped": int, "reason_<...>": int}
    """
    stats = defaultdict(int)
    rows: list[tuple[str,str,str]] = []

    symbol = infer_symbol_from_path(path)
    if not symbol:
        stats["reason_symbol_parse_fail"] += 1
        return rows, stats

    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            stats["reason_empty_file"] += 1
            return rows, stats

        cols = locate_cols(header)
        if cols["ts"] is None or cols["close"] is None:
            stats["reason_bad_header"] += 1
            return rows, stats

        for row in rdr:
            stats["read"] += 1
            if not row or len(row) <= max(cols["ts"], cols["close"]):
                stats["reason_short_row"] += 1
                continue
            ts = row[cols["ts"]].strip()
            px = row[cols["close"]].strip()
            if not ts or not px:
                stats["reason_missing_val"] += 1
                continue
            # 아주 가벼운 숫자 검증
            try:
                float(px)
            except Exception:
                stats["reason_non_numeric_price"] += 1
                continue
            rows.append((ts, symbol, px))
            stats["kept"] += 1

    stats["skipped"] = stats["read"] - stats["kept"]
    return rows, stats

def main():
    ap = argparse.ArgumentParser(description="Merge OHLCV csvs (3 folders, dedup) -> prices_hist.csv")
    ap.add_argument("--out", required=True, help="output prices csv (ts,symbol,price,src)")
    ap.add_argument("inputs", nargs="+", help="input glob(s), e.g. .\\cache_ohlcv\\*.csv .\\logs\\cache_tv\\*.csv")
    args = ap.parse_args()

    # 글롭 확장
    files: list[str] = []
    for pat in args.inputs:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    print(f"[scan] matched files: {len(files)}")
    if not files:
        print("No files matched. Double-check the paths / globs.")
        sys.exit(0)

    all_rows: dict[tuple[str,str], str] = {}  # (ts,symbol) -> price
    agg = defaultdict(int)

    for fp in files:
        rows, st = read_one_file(fp)
        for k, v in st.items():
            agg[k] += v

        for (ts, sym, px) in rows:
            key = (ts, sym)
            if key not in all_rows:
                all_rows[key] = px

    out_rows = [(ts, sym, px, "ohlcv") for (ts, sym), px in all_rows.items()]
    out_rows.sort(key=lambda x: (x[0], x[1]))

    # 저장
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "symbol", "price", "src"])
        w.writerows(out_rows)

    print(f"[done] written {len(out_rows)} rows -> {os.path.abspath(args.out)}")
    # 요약 통계
    if agg["read"] > 0:
        print(f"[stats] read={agg['read']}, kept={agg['kept']}, skipped={agg['skipped']}")
    skip_keys = [k for k in agg.keys() if k.startswith("reason_")]
    if skip_keys:
        print("[reasons]")
        for k in sorted(skip_keys):
            print(f"  {k}: {agg[k]}")

if __name__ == "__main__":
    main()