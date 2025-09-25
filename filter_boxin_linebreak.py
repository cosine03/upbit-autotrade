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
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

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

    # 1) timestamp normalize (LEFT)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()

    # 2) distance normalize -> ratio
    scale = detect_scale(df["distance_pct"])
    dist_ratio = pd.to_numeric(df["distance_pct"], errors="coerce")
    if scale == "percent":
        dist_ratio = dist_ratio / 100.0
    df["distance_ratio"] = dist_ratio

    # 3) price_in_box rows (RIGHT)
    level_cols = [c for c in ["level_id", "box_id", "line_id"] if c in df.columns]
    keep_cols = ["symbol", "ts"] + level_cols
    box_all = df.loc[df["event"] == "price_in_box", keep_cols].copy()

    if box_all.empty:
        df.head(0).to_csv(args.out, index=False, encoding="utf-8")
        print("[FILTER] No price_in_box found -> 0 rows written")
        return

    # 4) group-wise asof (symbol별 time-sort 준수)
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        b = box_all[box_all["symbol"] == sym].sort_values("ts").reset_index(drop=True)

        if b.empty:
            g["ts_last_box"] = pd.NaT
            for lc in level_cols:
                g[f"{lc}_last_box"] = pd.NA
        else:
            r = b.copy()
            r["ts_last_box"] = r["ts"]
            for lc in level_cols:
                r.rename(columns={lc: f"{lc}_last_box"}, inplace=True)

            # merge_asof은 같은 심볼 내 시간축만 필요
            g = pd.merge_asof(
                g, r.drop(columns=["symbol"]),
                on="ts", direction="backward"
            )

        parts.append(g)

    df2 = pd.concat(parts, ignore_index=True)

    # 5) RIGHT ts 캐스팅(강제) -> dt 연산 안전화
    df2["ts_last_box"] = pd.to_datetime(df2.get("ts_last_box"), utc=True, errors="coerce")

    # NaT 안전 계산: numpy timedelta로 변환 후 NaT는 큰 값으로 치환
    ts_ns   = df2["ts"].values.astype("datetime64[ns]")
    last_ns = df2["ts_last_box"].values.astype("datetime64[ns]")
    dt = (ts_ns - last_ns) / np.timedelta64(1, "s")   # float sec, NaT→nan
    dt = np.where(np.isnan(dt), np.inf, dt)           # NaT는 lookback 실패로 간주

    df2["box_lookback_ok"] = (dt <= args.lookback_hours * 3600)

    # 6) require same level (옵션)
    if args.require_same_level and level_cols:
        same_flags = []
        for lc in level_cols:
            last = f"{lc}_last_box"
            if last in df2.columns:
                same_flags.append(df2[lc].astype(str) == df2[last].astype(str))
        df2["same_level"] = np.logical_or.reduce(same_flags) if same_flags else False
        df2["box_lookback_ok"] &= df2["same_level"]
    else:
        df2["same_level"] = np.nan

    # 7) distance & event filter
    df2["dist_ok"] = (df2["distance_ratio"] >= 0) & (df2["distance_ratio"] <= args.dist_max)
    mask = (df2["event"] == "line_breakout") & df2["box_lookback_ok"] & df2["dist_ok"]
    out = df2.loc[mask].copy()

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[FILTER] scale={scale} | in={len(df)} -> out={len(out)} | "
          f"lookback_h={args.lookback_hours} dist_max={args.dist_max} | "
          f"require_same_level={args.require_same_level}")

if __name__ == "__main__":
    main()