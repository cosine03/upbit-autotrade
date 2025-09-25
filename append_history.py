# -*- coding: utf-8 -*-
import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--strategy", required=True, choices=["breakout_only","boxin_linebreak"])
    ap.add_argument("--date", required=True)
    ap.add_argument("--history", required=True, help="path to history csv to append to")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    keep = ["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"]
    df = df[keep].copy()
    df["strategy"] = args.strategy; df["date"] = args.date

    if os.path.exists(args.history):
        old = pd.read_csv(args.history)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out.to_csv(args.history, index=False, encoding="utf-8")
    print("[HISTORY] appended ->", args.history, "| rows:", len(out))

if __name__ == "__main__":
    main()