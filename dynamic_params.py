# -*- coding: utf-8 -*-
"""
Dynamic recommender for (1) dist_max and (2) TP/SL per event.

Usage example:
python -u dynamic_params.py --signals .\logs\signals_tv_enriched.csv ^
  --grid-csv .\logs\grid_tp_sl_dm00025\grid_summary_dm00025.csv ^
  --target-lo 20 --target-hi 30 --include-zero ^
  --events box_breakout,line_breakout,price_in_box ^
  --min-trades 30 --metric avg_net ^
  --out .\logs\dynamic_params.json
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def detect_scale(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not len(s):
        return "ratio"
    return "ratio" if float(s.max()) <= 1.0 else "percent"


def pick_dist_by_count(df: pd.DataFrame, target_lo: int, target_hi: int,
                       include_zero: bool = True,
                       clamp_min: float = 0.0, clamp_max: float = 0.001) -> dict:
    if "distance_pct" not in df.columns:
        raise ValueError("distance_pct missing in signals file")
    s = pd.to_numeric(df["distance_pct"], errors="coerce")
    scale = detect_scale(s)
    s = s / 100.0 if scale == "percent" else s

    pool = s.dropna().copy()
    if not include_zero:
        pool = pool[pool > 0]

    if len(pool) == 0:
        thr = 0.0
        cnt = 0
    else:
        vals = pool.sort_values().reset_index(drop=True)
        i_lo = max(0, min(target_lo - 1, len(vals) - 1))
        i_hi = max(0, min(target_hi - 1, len(vals) - 1))
        thr = float((vals.iloc[i_lo] + vals.iloc[i_hi]) / 2.0)
        cnt = int((s <= thr).sum()) if include_zero else int(((s > 0) & (s <= thr)).sum())

    # clamp for safety
    thr = max(clamp_min, min(thr, clamp_max))
    return {
        "method": "target_count",
        "scale": scale,
        "include_zero": bool(include_zero),
        "threshold_ratio": float(thr),
        "threshold_units": float(thr if scale == "ratio" else thr * 100.0),
        "est_count": int(cnt),
        "n": int(len(s.dropna())),
        "clamp": {"min": clamp_min, "max": clamp_max},
    }


def pick_dist_by_percentile(df: pd.DataFrame, pct: float,
                            include_zero: bool = True,
                            clamp_min: float = 0.0, clamp_max: float = 0.001) -> dict:
    if "distance_pct" not in df.columns:
        raise ValueError("distance_pct missing in signals file")
    s = pd.to_numeric(df["distance_pct"], errors="coerce")
    scale = detect_scale(s)
    s = s / 100.0 if scale == "percent" else s

    pool = s.dropna().copy()
    if not include_zero:
        pool = pool[pool > 0]

    if len(pool) == 0:
        thr = 0.0
    else:
        thr = float(np.quantile(pool.values, pct))

    thr = max(clamp_min, min(thr, clamp_max))
    cnt = int((s <= thr).sum()) if include_zero else int(((s > 0) & (s <= thr)).sum())
    return {
        "method": "percentile",
        "scale": scale,
        "include_zero": bool(include_zero),
        "threshold_ratio": float(thr),
        "threshold_units": float(thr if scale == "ratio" else thr * 100.0),
        "est_count": int(cnt),
        "n": int(len(s.dropna())),
        "pct": float(pct),
        "clamp": {"min": clamp_min, "max": clamp_max},
    }


def load_grid_csvs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score_tp_sl(grid_df: pd.DataFrame, min_trades: int = 30,
                metric: str = "avg_net") -> pd.DataFrame:
    if grid_df.empty:
        return grid_df

    g = (grid_df.groupby(["event", "tp", "sl"], dropna=False)
         .agg(trades=("trades", "sum"),
              avg_net_mean=("avg_net", "mean"),
              total_net_sum=("total_net", "sum"),
              win_rate_mean=("win_rate", "mean"))
         .reset_index())

    g = g[g["trades"] >= int(min_trades)].copy()

    if metric == "total_net":
        g["score"] = g["total_net_sum"]
    else:
        g["score"] = g["avg_net_mean"]

    g["score"] = g["score"] + 0.0005 * (g["win_rate_mean"] - 0.5) + 1e-6 * g["trades"]
    g = g.sort_values(["event", "score"], ascending=[True, False]).reset_index(drop=True)
    return g


def select_top_per_event(scored: pd.DataFrame, events: list[str], top_k: int = 1) -> pd.DataFrame:
    out = []
    for ev in events:
        gg = scored[scored["event"] == ev]
        if gg.empty:
            continue
        out.append(gg.head(top_k).assign(event=ev))
    return pd.concat(out, ignore_index=True) if out else scored.iloc[0:0].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, help="today's signals csv (for dist calc)")
    ap.add_argument("--grid-csv", nargs="*", default=[], help="paths to past grid_summary_*.csv")
    ap.add_argument("--target-lo", type=int, default=20)
    ap.add_argument("--target-hi", type=int, default=30)
    ap.add_argument("--include-zero", action="store_true", help="include 0.0 distance in dist selection")
    ap.add_argument("--use-percentile", type=float, default=None, help="if set (0..1), use percentile for dist")
    ap.add_argument("--clamp-min", type=float, default=0.0, help="min dist_max (ratio)")
    ap.add_argument("--clamp-max", type=float, default=0.001, help="max dist_max (ratio)")
    ap.add_argument("--min-trades", type=int, default=30, help="min trades per (event,tp,sl) to be eligible")
    ap.add_argument("--metric", default="avg_net", choices=["avg_net", "total_net"])
    ap.add_argument("--events", default="box_breakout,line_breakout,price_in_box", help="comma list")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--out", default="./logs/dynamic_params.json")
    args = ap.parse_args()

    sig = pd.read_csv(args.signals)
    if args.use_percentile is not None:
        dist_res = pick_dist_by_percentile(sig, pct=float(args.use_percentile),
                                           include_zero=args.include_zero,
                                           clamp_min=args.clamp_min, clamp_max=args.clamp_max)
    else:
        dist_res = pick_dist_by_count(sig, args.target_lo, args.target_hi,
                                      include_zero=args.include_zero,
                                      clamp_min=args.clamp_min, clamp_max=args.clamp_max)

    events = [e.strip() for e in args.events.split(",") if e.strip()]
    grid = load_grid_csvs(args.grid_csv)
    if not grid.empty:
        grid = grid[grid["event"].isin(events)].copy()
    scored = score_tp_sl(grid, min_trades=args.min_trades, metric=args.metric) if not grid.empty else pd.DataFrame()
    top = select_top_per_event(scored, events, top_k=args.top_k) if not scored.empty else pd.DataFrame()

    payload = {
        "dist_max": dist_res,
        "tp_sl": top.to_dict(orient="records") if not top.empty else [],
        "notes": {
            "tp_sl_source": "aggregated from provided grid CSVs" if not grid.empty else "no grid CSVs provided",
            "dist_logic": "percentile" if args.use_percentile is not None else "target_count",
            "ranges": {
                "dist_clamp": [args.clamp_min, args.clamp_max],
                "tp_list_hint": "as per grid CSVs previously run",
                "events_considered": events,
            },
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
