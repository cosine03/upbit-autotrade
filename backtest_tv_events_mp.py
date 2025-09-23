import argparse, os, time
import pandas as pd
import numpy as np
import multiprocessing as mp

# -----------------------------
# Utility functions
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("signals", help="signals_tv.csv (or enriched csv)")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--expiries", default="4h,8h", help="Comma-separated list, e.g. 2h,4h,8h")
    p.add_argument("--tp", type=float, default=1.5)
    p.add_argument("--sl", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--dist-max", type=float, default=None, help="max abs distance_pct filter")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--outdir", default="./logs")
    return p.parse_args()

def ensure_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def ts_to_ns(s: pd.Series):
    """Convert to numpy datetime64[ns] UTC-naive"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize(None).astype("datetime64[ns]").to_numpy()
    else:
        return pd.to_datetime(s, utc=True, errors="coerce").tz_localize(None).astype("datetime64[ns]").to_numpy()

# -----------------------------
# Core simulation
# -----------------------------
def simulate_symbol(symbol, df_sig, timeframe, tp, sl, fee_rt, expiries_h):
    # 여기서는 간단화: 실제 OHLCV 로딩은 더미 처리 (실전에서는 API/DB 필요)
    trades = []
    ts_arr = ts_to_ns(df_sig["ts"])

    for i, row in df_sig.iterrows():
        sig_ts = pd.to_datetime(row["ts"], utc=True).tz_localize(None)
        for eh in expiries_h:
            # 더미: 랜덤 결과 (실전에서는 OHLCV 기반 진입/청산 로직 필요)
            result = np.random.choice([-1, 1])
            net = (tp - fee_rt) if result > 0 else -(sl + fee_rt)
            trades.append({
                "symbol": symbol,
                "ts": sig_ts,
                "event": row["event"],
                "group": row["event_group"],
                "expiry_h": eh,
                "tp": tp, "sl": sl,
                "net": net,
                "win": 1 if net > 0 else 0
            })
    return pd.DataFrame(trades)

def run_group(df, group, timeframe, tp, sl, fee_rt, expiries_h, procs):
    df_g = df[df["event_group"] == group]
    symbols = df_g["symbol"].unique().tolist()
    tasks = [(sym, df_g[df_g["symbol"] == sym], timeframe, tp, sl, fee_rt, expiries_h) for sym in symbols]

    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    with mp.Pool(processes=procs) as pool:
        parts = pool.starmap(simulate_symbol, tasks)

    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def agg(gr):
        return pd.Series({
            "trades": len(gr),
            "win_rate": gr["win"].mean(),
            "avg_net": gr["net"].mean(),
            "median_net": gr["net"].median(),
            "total_net": gr["net"].sum(),
        })
    stats = trades.groupby(["group", "expiry_h"]).apply(agg).reset_index()

    return trades, stats

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    ensure_outdir(args.outdir)
    df = pd.read_csv(args.signals)

    # step1: distance_pct 필터
    if args.dist_max is not None and "distance_pct" in df.columns:
        before = len(df)
        df = df[df["distance_pct"].abs() <= args.dist_max].copy()
        print(f"[BT] distance_pct filter {args.dist_max}: {before}->{len(df)} rows")

    # step2: 중복 제거 (line vs box)
    if "event" in df.columns:
        df["event_group"] = df["event"].replace({
            "box_breakout": "box_breakout",
            "line_breakout": "line_breakout",
            "price_in_box": "price_in_box",
            "level2_detected": "detected",
            "level3_detected": "detected",
        })
        dup_keys = df[df["event_group"].isin(["box_breakout","line_breakout"])]
        df = df.sort_values(["symbol","ts","event_group"])
        df = df.drop_duplicates(subset=["symbol","ts"], keep="last")

    expiries_h = [float(e.strip("h")) for e in args.expiries.split(",")]
    all_trades, all_stats = [], []

    for grp in ["detected","price_in_box","box_breakout","line_breakout"]:
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl, args.fee, expiries_h, args.procs)
        if not tr.empty:
            trades_path = os.path.join(args.outdir, f"bt_tv_events_trades_{grp}.csv")
            stats_path = os.path.join(args.outdir, f"bt_tv_events_stats_{grp}.csv")
            tr.to_csv(trades_path, index=False)
            st.to_csv(stats_path, index=False)
            print(f"[BT][{grp}] trades -> {trades_path} (rows={len(tr)})")
            print(f"[BT][{grp}] stats  -> {stats_path} (rows={len(st)})")
            all_trades.append(tr)
            all_stats.append(st)

    if all_trades:
        trades = pd.concat(all_trades, ignore_index=True)
        stats = pd.concat(all_stats, ignore_index=True)
        trades.to_csv(os.path.join(args.outdir, "bt_tv_events_trades.csv"), index=False)
        stats.to_csv(os.path.join(args.outdir, "bt_tv_events_stats.csv"), index=False)
        print("\n=== Summary (by event group & expiry) ===")
        print(stats)

if __name__ == "__main__":
    main()