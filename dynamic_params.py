# -*- coding: utf-8 -*-
"""
Estimate dynamic dist_max and pick TP/SL per event from grid summary (robust to column names).
Outputs JSON with dist_max.value_ratio/value_percent for pipeline auto-apply.
"""
import argparse, json
import numpy as np
import pandas as pd

def estimate_dist(signals_csv, include_zero, target_lo, target_hi, clamp_min, clamp_max):
    df = pd.read_csv(signals_csv)
    if "distance_pct" not in df.columns:
        raise SystemExit("distance_pct missing in signals file")
    x = pd.to_numeric(df["distance_pct"], errors="coerce").dropna().values
    scale = "ratio" if (len(x)==0 or np.nanmax(x) <= 1.0) else "percent"
    x_ratio = x/100.0 if scale=="percent" else x
    if not include_zero:
        x_ratio = x_ratio[x_ratio > 0]
    x_sorted = np.sort(x_ratio)
    n = len(x_sorted)
    if n == 0:
        thr = clamp_max
        est_count = 0
    else:
        k = max(1, min(target_lo, n)) - 1
        thr = float(x_sorted[k])
        thr = float(np.clip(thr, clamp_min, clamp_max))
        est_count = int(np.sum(x_ratio <= thr))
    return {
        "method": "target_count",
        "scale": scale,
        "include_zero": bool(include_zero),
        "threshold_ratio": 0.0,
        "threshold_units": float(thr),
        "value_ratio": float(thr),
        "value_percent": float(thr*100.0),
        "est_count": int(est_count),
        "n": int(n),
        "clamp": {"min": float(clamp_min), "max": float(clamp_max)}
    }

def _pick_col(df, *names, default=None):
    for n in names:
        if n in df.columns:
            return n
    return default

def normalize_grid(g):
    """
    Returns aggregated grid with columns:
      event, tp, sl, trades, avg_net_mean, win_rate_mean, total_net_sum
    Accepts raw or pre-aggregated CSVs with various column names.
    """
    g = g.copy()
    # Harmonize column names if present
    c_ev = _pick_col(g, "event", "Event", "EVENT")
    c_tp = _pick_col(g, "tp", "TP")
    c_sl = _pick_col(g, "sl", "SL")
    c_tr = _pick_col(g, "trades", "n_trades", "count", "N", default=None)
    c_avg = _pick_col(g, "avg_net_mean", "avg_net_avg", "avg_net", "avg", default=None)
    c_wr  = _pick_col(g, "win_rate_mean", "win_rate_avg", "win_rate", "wr", default=None)
    c_tot = _pick_col(g, "total_net_sum", "total_net", "sum_net", default=None)

    need_agg = any(c is None for c in [c_tr, c_avg, c_wr, c_tot])
    # If key columns missing, try canonical names
    if c_ev is None or c_tp is None or c_sl is None:
        raise SystemExit("grid CSV must contain event/tp/sl columns")

    if not need_agg:
        out = g.rename(columns={
            c_tr: "trades",
            c_avg: "avg_net_mean",
            c_wr:  "win_rate_mean",
            c_tot: "total_net_sum",
            c_ev:  "event",
            c_tp:  "tp",
            c_sl:  "sl",
        }).copy()
        return out[["event","tp","sl","trades","avg_net_mean","win_rate_mean","total_net_sum"]]

    # Need to aggregate from raw columns; try to find raw metrics
    # Prefer: per-trade 'net' then compute mean/sum + win rate
    c_net = _pick_col(g, "net", "avg_net")  # sometimes only avg exists; fall back
    c_win = _pick_col(g, "win", "is_win", "won")  # boolean/int
    gg = g.copy()
    # Basic conversions
    if c_win in gg.columns:
        wr_series = pd.to_numeric(gg[c_win], errors="coerce")
        # normalize to 0/1
        wr_series = (wr_series > 0).astype(float)
        gg["_win"] = wr_series
    else:
        gg["_win"] = np.nan

    if c_net in gg.columns:
        gg["_net"] = pd.to_numeric(gg[c_net], errors="coerce")
    else:
        gg["_net"] = np.nan

    out = gg.groupby([c_ev, c_tp, c_sl], dropna=False).agg(
        trades=("event","count"),
        avg_net_mean=("_net","mean"),
        win_rate_mean=("_win","mean"),
        total_net_sum=("_net","sum"),
    ).reset_index().rename(columns={c_ev:"event", c_tp:"tp", c_sl:"sl"})

    # if win_rate_mean was NaN, try to backfill from any available win_rate columns
    if out["win_rate_mean"].isna().all() and c_wr is not None:
        # merge one-shot average from original g
        wr_by = gg.groupby([c_ev, c_tp, c_sl], dropna=False)[c_wr].mean().reset_index()
        out = out.merge(
            wr_by.rename(columns={c_ev:"event", c_tp:"tp", c_sl:"sl", c_wr:"win_rate_mean"}),
            on=["event","tp","sl"], how="left", suffixes=("","_alt")
        )
        out["win_rate_mean"] = out["win_rate_mean"].fillna(out.pop("win_rate_mean_alt"))

    # NaNs to zeros for scoring safety
    for c in ["avg_net_mean","win_rate_mean","total_net_sum"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out

def pick_tp_sl(grid_csv, events):
    raw = pd.read_csv(grid_csv)
    g = normalize_grid(raw)
    out = []
    for ev in [e for e in events.split(",") if e]:
        sub = g[g["event"] == ev]
        if sub.empty:
            continue
        # score: prioritize avg_net_mean, tiny tie-breaker with win_rate
        sc = sub["avg_net_mean"] + 1e-4 * sub["win_rate_mean"]
        k = int(sc.idxmax())
        row = sub.loc[k]
        out.append({
            "event": ev,
            "tp": float(row["tp"]),
            "sl": float(row["sl"]),
            "trades": int(row.get("trades", 0)),
            "avg_net_mean": float(row.get("avg_net_mean", 0.0)),
            "total_net_sum": float(row.get("total_net_sum", 0.0)),
            "win_rate_mean": float(row.get("win_rate_mean", 0.0)),
            "score": float(sc.loc[k]),
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
            "tp_sl_source": "grid CSV (robust columns)",
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