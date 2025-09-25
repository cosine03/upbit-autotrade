# -*- coding: utf-8 -*-
r"""
Filter: keep only line_breakout signals that had a recent price_in_box (same symbol),
and within a distance threshold.

Usage:
python -u filter_boxin_linebreak.py .\logs\signals_tv_enriched.csv ^
  --out .\logs\signals_boxin_linebreak.csv ^
  --lookback-hours 48 ^
  --dist-max 0.00025 ^
  [--require-same-level]

Notes:
- dist_max expects RATIO scale (e.g., 0.00025 = 0.025%). If your distance_pct is in 0..100,
  the script auto-detects and converts.
"""
from __future__ import annotations

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
    ap.add_argument("--dist-max", type=float, default=0.00025, help="RATIO (0.00025=0.025%)")
    ap.add_argument("--require-same-level", action="store_true",
                    help="if *_id columns exist, require same level between box and breakout")
    args = ap.parse_args()

    df = pd.read_csv(args.signals)
    for col in ["event", "symbol", "ts", "distance_pct"]:
        if col not in df.columns:
            raise SystemExit(f"Required column missing: {col}")

    # timestamp normalize
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()

    # distance normalize -> ratio
    scale = detect_scale(df["distance_pct"])
    dist_ratio = pd.to_numeric(df["distance_pct"], errors="coerce")
    if scale == "percent":
        dist_ratio = dist_ratio / 100.0
    df["distance_ratio"] = dist_ratio

    # prepare price_in_box rows (keep optional id columns)
    level_cols = [c for c in ["level_id", "box_id", "line_id"] if c in df.columns]
    keep_cols = ["symbol", "ts"] + level_cols
    box_all = df.loc[df["event"] == "price_in_box", keep_cols].copy()

    # fast exit if no box
    if box_all.empty:
        pd.DataFrame(columns=df.columns).to_csv(args.out, index=False, encoding="utf-8")
        print("[FILTER] No price_in_box found -> 0 rows written")
        return

    # group-wise asof merge (per symbol)
    groups = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        box_g = box_all[box_all["symbol"] == sym].sort_values("ts").reset_index(drop=True)

        if box_g.empty:
            g["ts_last_box"] = pd.NaT
            for lc in level_cols:
                g[f"{lc}_last_box"] = pd.NA
        else:
            # right side: keep ts for join + duplicate as ts_last_box + suffix level cols
            box_r = box_g.copy()
            box_r["ts_last_box"] = box_r["ts"]
            for lc in level_cols:
                box_r.rename(columns={lc: f"{lc}_last_box"}, inplace=True)

            merged = pd.merge_asof(
                g, box_r.drop(columns=["symbol"]),  # same symbol already
                on="ts", direction="backward"
            )
            g = merged

        groups.append(g)

    df2 = pd.concat(groups, ignore_index=True)

    # within lookback?
    dt_sec = (df2["ts"] - df2["ts_last_box"]).dt.total_seconds()
    df2["box_lookback_ok"] = (df2["ts_last_box"].notna()) & (dt_sec <= args.lookback_hours * 3600)

    # require same level if requested
    if args.require-same-level and level_cols:
        same_flags = []
        for lc in level_cols:
            last = f"{lc}_last_box"
            if last in df2.columns:
                same_flags.append(df2[lc].astype(str) == df2[last].astype(str))
        df2["same_level"] = np.logical_or.reduce(same_flags) if same_flags else False
        df2["box_lookback_ok"] &= df2["same_level"]
    else:
        df2["same_level"] = np.nan

    # distance filter & final event pick
    df2["dist_ok"] = (df2["distance_ratio"] >= 0) & (df2["distance_ratio"] <= args.dist_max)
    mask = (df2["event"] == "line_breakout") & df2["box_lookback_ok"] & df2["dist_ok"]
    out = df2.loc[mask].copy()

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[FILTER] scale={scale} | in={len(df)} -> out={len(out)} | "
          f"lookback_h={args.lookback_hours} dist_max={args.dist_max} "
          f"| require_same_level={args.require_same_level}")

if __name__ == "__main__":
    main()