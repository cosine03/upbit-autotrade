# -*- coding: utf-8 -*-
import argparse, re
import pandas as pd
import numpy as np

SUPPORT_KEYS = ["support", "sup"]
RESIST_KEYS  = ["resist", "resistance", "line_breakout", "box_breakout", "breakout"]

def infer_side(ev: str) -> str:
    e = (ev or "").lower()
    if any(k in e for k in SUPPORT_KEYS):
        return "support"
    if any(k in e for k in RESIST_KEYS):
        return "resistance"
    return "neutral"

def extract_touch_n(ev: str):
    if not isinstance(ev, str): return np.nan
    m = re.search(r"(\d+)", ev)
    return int(m.group(1)) if m else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="signals csv (tv_enriched)")
    ap.add_argument("--out", required=True, help="labeled csv path")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    if "event" not in df.columns:
        raise SystemExit("input csv must contain 'event' column")

    # timestamp 정규화
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    df["event_str"] = df["event"].astype(str)
    df["side"] = df["event_str"].map(infer_side)
    df["is_support"] = df["side"].eq("support")
    df["is_resistance"] = df["side"].eq("resistance")
    df["touch_n"] = df["event_str"].map(extract_touch_n)

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[LABEL] in={len(df)} -> out={args.out} | side counts=", df["side"].value_counts().to_dict())

if __name__ == "__main__":
    main()