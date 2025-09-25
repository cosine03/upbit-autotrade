# -*- coding: utf-8 -*-
"""
Estimate dynamic dist_max and pick TP/SL per event from grid summary.
Outputs JSON with dist_max.value_ratio/value_percent for pipeline auto-apply.
"""
import argparse, json, pandas as pd, numpy as np

def estimate_dist(signals_csv, include_zero, target_lo, target_hi, clamp_min, clamp_max):
    df = pd.read_csv(signals_csv)
    if "distance_pct" not in df.columns:
        raise SystemExit("distance_pct missing")
    x = pd.to_numeric(df["distance_pct"], errors="coerce").dropna().values
    # auto scale detect
    scale = "ratio" if np.nanmax(x) <= 1.0 else "percent"
    if scale == "percent":
        x_ratio = x / 100.0
    else:
        x_ratio = x

    if include_zero:
        x_ratio = x_ratio
    else:
        x_ratio = x_ratio[x_ratio > 0]

    x_sorted = np.sort(x_ratio)
    n = len(x_sorted)

    # simple search for threshold s.t. count in [target_lo, target_hi]
    # pick the smallest value achieving target_lo; clamp afterward
    thr = clamp_max if n == 0 else (x_sorted[target_lo-1] if n >= target_lo else x_sorted[-1] if n>0 else clamp_max)
    thr = float(np.clip(thr, clamp_min, clamp_max))
    est_count = int(np.sum(x_ratio <= thr))
    return {
        "method": "target_count",
        "scale": scale,
        "include_zero": bool(include_zero),
        "threshold_ratio": 0.0,
        "threshold_units": float(thr),
        "value_ratio": float(thr),                       # <- pipeline uses this
        "value_percent": float(thr*100.0),               # convenience
        "est_count": int(est_count),
        "n": int(n),
        "clamp": {"min": float(clamp_min), "max": float(clamp_max)}
    }

def pick_tp_sl(grid_csv, events):
    g = pd.read_csv(grid_csv)
    # expect cols: event, tp, sl, trades, avg_net_mean, total_net_sum, win_rate_mean
    out = []
    for ev in events.split(","):
        sub = g[g["event"]==ev]
        if sub.empty: 
            continue
        # score 예시: avg_net_mean(우선) + 약간의 승률 보정
        sc = sub["avg_net_mean"] + 0.0001*sub["win_rate_mean"]
        k  = int(sc.idxmax())
        row = sub.loc[k]
        out.append({
            "event": ev,
            "tp": float(row["tp"]),
            "sl": float(row["sl"]),
            "trades": int(row.get("trades", 0)),
            "avg_net_mean": float(row.get("avg_net_mean", 0.0)),
            "total_net_sum": float(row.get("total_net_sum", 0.0)),
            "win_rate_mean": float(row.get("win_rate_mean", 0.0)),
            "score": float(sc.loc[k])
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--grid-csv", required=True)
    ap.add_argument("--target-lo", type=int, default=20)
    ap.add_argument("--target-hi", type=int, default=30)
    ap.add_argument("--include-zero", action="store_true")
    ap.add_argument("--events", default="box_breakout,line_breakout,price_in_box")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--metric", default="avg_net")
    ap.add_argument("--clamp-min", type=float, default=0.0)
    ap.add_argument("--clamp-max", type=float, default=0.001)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dist = estimate_dist(args.signals, args.include_zero, args.target_lo, args.target_hi, args.clamp_min, args.clamp_max)
    tp_sl = pick_tp_sl(args.grid_csv, args.events)

    out = {
        "dist_max": dist,
        "tp_sl": tp_sl,
        "notes": {
            "tp_sl_source": "aggregated from provided grid CSVs",
            "dist_logic": "target_count",
            "ranges": {
                "dist_clamp": [args.clamp_min, args.clamp_max],
                "events_considered": args.events.split(",")
            }
        }
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()