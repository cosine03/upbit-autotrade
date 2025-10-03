#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, sys, argparse, os, re
from datetime import datetime, timezone

TARGET_FIELDS = [
    "ts","event","side","level","touches","symbol",
    "timeframe","extra","host","message","distance_pct"
]
DIST_ZERO = {"line_breakout","box_breakout","price_in_box"}

def norm_row(row):
    out = {k:"" for k in TARGET_FIELDS}
    # 컬럼 매핑(이름 다를 수 있는 것 처리)
    # 기본적으로 key가 있으면 그대로, 없으면 공란
    for k in list(row.keys()):
        row[k] = row[k].strip() if isinstance(row[k], str) else row[k]

    out["ts"]        = row.get("ts","")
    out["event"]     = row.get("event","")
    out["side"]      = row.get("side","")
    out["level"]     = row.get("level","")
    out["touches"]   = row.get("touches","")
    out["symbol"]    = row.get("symbol","")
    out["timeframe"] = row.get("timeframe","")
    out["extra"]     = row.get("extra","")
    out["host"]      = row.get("host","")
    out["message"]   = row.get("message","")

    dp = row.get("distance_pct","")
    if dp == "" and out["event"] in DIST_ZERO:
        dp = "0.0"  # 과거 파일에도 동일 규칙 적용
    out["distance_pct"] = dp

    return out

def parse_ts(s):
    # 2025-09-30T12:15:30+00:00 형태 가정
    try:
        return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Merge signals_tv (bak + current) into a snapshot")
    ap.add_argument("--out", required=False, default=None, help="output csv (default: logs/signals_tv_snapshot_*.csv)")
    ap.add_argument("inputs", nargs="+", help="input files (e.g. logs/signals_tv_*.bak logs/signals_tv.csv)")
    args = ap.parse_args()

    rows = []
    seen = set()  # 중복 제거 키
    for path in args.inputs:
        if not os.path.exists(path): 
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                n = norm_row(row)
                key = (n["ts"], n["event"], n["side"], n["level"], n["touches"], n["symbol"], n["message"])
                if key in seen: 
                    continue
                seen.add(key)
                rows.append(n)

    # ts 기준 정렬
    rows.sort(key=lambda x: (parse_ts(x["ts"]) or datetime.min.replace(tzinfo=timezone.utc), x["symbol"], x["event"]))

    # 아웃 경로
    if not args.out:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = os.path.join(".", "logs", f"signals_tv_snapshot_{ts}.csv")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TARGET_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"snapshot written: {os.path.abspath(args.out)}")
    print(f"rows: {len(rows)}")

if __name__ == "__main__":
    main()