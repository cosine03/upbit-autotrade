# -*- coding: utf-8 -*-
"""
Pre-filter: keep only line_breakout signals that had a recent price_in_box
for the same symbol (optionally same level_id if present), and within a distance threshold.

Usage:
python -u filter_boxin_linebreak.py .\logs\signals_tv_enriched.csv ^
  --out .\logs\signals_boxin_linebreak.csv ^
  --lookback-hours 48 ^
  --dist-max 0.00025 ^
  --require-same-level
"""
import argparse
import pandas as pd
import numpy as np


def detect_scale(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not len(s):
        return "ratio"
    return "ratio" if float(s.max()) <= 1.0 else "percent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv_enriched.csv path")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--lookback-hours", type=int, default=48)
    ap.add_argument("--dist-max", type=float, default=0.00025,
                    help="RATIO scale (e.g., 0.00025 = 0.025%)")
    ap.add_argument("--require-same-level", action="store_true",
                    help="if level_id columns exist, require same level between box and breakout")
    args = ap.parse_args()

    df = pd.read_csv(args.signals)
    if "event" not in df.columns or "symbol" not in df.columns:
        raise SystemExit("Required columns missing: event, symbol")
    if "ts" not in df.columns:
        raise SystemExit("Required column missing: ts")

    # normalize timestamp
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()

    # normalize distance to ratio
    if "distance_pct" not in df.columns:
        raise SystemExit("Required column missing: distance_pct")
    scale = detect_scale(df["distance_pct"])
    dist_ratio = pd.to_numeric(df["distance_pct"], errors="coerce")
    if scale == "percent":
        dist_ratio = dist_ratio / 100.0
    df["distance_ratio"] = dist_ratio

    # gather price_in_box events
    keep_cols = ["symbol", "ts"]
    level_cols = []
    for c in ["level_id", "box_id", "line_id"]:
        if c in df.columns:
            level_cols.append(c)
    keep_cols += level_cols

    box = df.loc[df["event"] == "price_in_box", keep_cols].copy()
    box = box.sort_values(["symbol", "ts"]).reset_index(drop=True)
    if box.empty:
        df.iloc[0:0].to_csv(args.out, index=False, encoding="utf-8")
        print("[FILTER] No price_in_box in input -> 0 rows written")
        return

    # asof join: attach last box ts per symbol
    df_sorted = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    box_sorted = box.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df2 = pd.merge_asof(
        df_sorted, box_sorted,
        on="ts", by="symbol", direction="backward",
        suffixes=("", "_last_box")
    )

    # lookback check
    dt_sec = (df2["ts"] - df2["ts_last_box"]).dt.total_seconds()
    df2["box_lookback_ok"] = (df2["ts_last_box"].notna()) & (dt_sec <= args.lookback_hours * 3600)

    # same-level check (optional)
    if args.require_same_level and level_cols:
        same_flags = []
        for c in level_cols:
            last_c = f"{c}_last_box"
            if c in df2.columns and last_c in df2.columns:
                same_flags.append(df2[c].astype(str) == df2[last_c].astype(str))
        df2["same_level"] = np.logical_or.reduce(same_flags) if same_flags else False
        df2["box_lookback_ok"] &= df2["same_level"]
    else:
        df2["same_level"] = np.nan

    # distance check
    df2["dist_ok"] = (df2["distance_ratio"] >= 0) & (df2["distance_ratio"] <= args.dist_max)

    # final mask
    mask = (df2["event"] == "line_breakout") & df2["box_lookback_ok"] & df2["dist_ok"]
    out = df2.loc[mask].copy()

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[FILTER] scale={scale} | in={len(df)} -> out={len(out)} "
          f"| lookback_h={args.lookback_hours} dist_max={args.dist_max} "
          f"| require_same_level={args.require_same_level}")


if __name__ == "__main__":
    main()