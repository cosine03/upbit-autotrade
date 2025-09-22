# backtest_upbit_multi_strategies_mp.py
# -*- coding: utf-8 -*-
import os, argparse
from multiprocessing import Pool, cpu_count
from datetime import timedelta
import numpy as np
import pandas as pd

from sr_engine.data import get_ohlcv

UPBIT_PATH_DEFAULT = "./logs/signals_upbit.csv"
CACHE_DIR = "./logs/cache_upbit"
os.makedirs(CACHE_DIR, exist_ok=True)

MAJOR = {"KRW-BTC", "KRW-ETH"}

STRATS = [
    ("stable_1.5/1.0",    0.015,  0.010),
    ("aggressive_2.0/1.25",0.020,  0.0125),
    ("scalp_1.0/0.75",     0.010,  0.0075),
    ("mid_1.25/1.0",       0.0125, 0.010),
    ("mid_1.75/1.25",      0.0175, 0.0125),
    ("tight_0.8/0.8",      0.008,  0.008),
]

def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["ts","symbol","event","side","level","timeframe","message"]
    for c in need:
        if c not in df.columns:
            df[c] = ""
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df

def filter_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "major":
        return df[df["symbol"].isin(MAJOR)].copy()
    if group == "alt":
        return df[~df["symbol"].isin(MAJOR)].copy()
    return df

def get_ohlcv_cached(symbol: str, timeframe: str) -> pd.DataFrame:
    p = os.path.join(CACHE_DIR, f"{symbol}_{timeframe}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    df = get_ohlcv(symbol, timeframe)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            df = df.reset_index(drop=True)
            df["ts"] = ts
        else:
            df["ts"] = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy().sort_values("ts")
    df.to_csv(p, index=False)
    return df

def next_bar_open_idx(ohlcv: pd.DataFrame, sig_ts_utc: pd.Timestamp) -> int:
    ts = ohlcv["ts"].to_numpy()
    return int(np.searchsorted(ts, sig_ts_utc.to_datetime64(), side="right"))

def _fees_roundtrip(pct: float) -> float:
    return pct - 0.001  # 왕복 0.1%

def simulate_symbol(args_sym):
    symbol, rows, timeframe, horizon_h = args_sym
    ohlcv = get_ohlcv_cached(symbol, timeframe)
    ohlcv = ohlcv.sort_values("ts").reset_index(drop=True)
    out_all = []

    for _, s in rows.iterrows():
        sig_ts = pd.Timestamp(s["ts"]).tz_convert("UTC") if pd.Timestamp(s["ts"]).tzinfo else pd.Timestamp(s["ts"]).tz_localize("UTC")
        i_enter = next_bar_open_idx(ohlcv, sig_ts)
        if i_enter >= len(ohlcv):
            continue
        px_enter = float(ohlcv["open"].iloc[i_enter])

        dt_enter = pd.Timestamp(ohlcv["ts"].iloc[i_enter]).to_pydatetime()
        dt_expiry = dt_enter + timedelta(hours=horizon_h)

        idx_exp = int(np.searchsorted(ohlcv["ts"].to_numpy(), np.datetime64(pd.Timestamp(dt_expiry, tz="UTC")), side="left"))
        if idx_exp >= len(ohlcv):
            idx_exp = len(ohlcv)-1

        # 각 전략 TP/SL 체크 → 조기청산 or 만기청산
        for name, tp, sl in STRATS:
            exit_idx = idx_exp
            outcome = None

            # i_enter+1부터 만기까지 고/저 스윕
            for i in range(i_enter+1, idx_exp+1):
                high_i = float(ohlcv["high"].iloc[i])
                low_i  = float(ohlcv["low"].iloc[i])
                # TP 먼저
                if (high_i / px_enter - 1.0) >= tp:
                    exit_idx = i
                    outcome = "TP"
                    break
                # SL
                if (low_i / px_enter - 1.0) <= -sl:
                    exit_idx = i
                    outcome = "SL"
                    break

            px_exit = float(ohlcv["close"].iloc[exit_idx])
            ret = (px_exit / px_enter) - 1.0
            ret = _fees_roundtrip(ret)

            out_all.append({
                "strategy": name,
                "symbol": symbol,
                "signal_ts": sig_ts.isoformat(),
                "enter_ts": pd.Timestamp(ohlcv["ts"].iloc[i_enter]).isoformat(),
                "exit_ts": pd.Timestamp(ohlcv["ts"].iloc[exit_idx]).isoformat(),
                "enter_px": px_enter,
                "exit_px": px_exit,
                "net": ret,
                "result": outcome or "EXPIRY",
            })

    return pd.DataFrame(out_all)

def agg_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"])
    g = df.groupby("strategy")
    res = g["net"].agg(["count","mean","median","sum"]).reset_index()
    res.columns = ["strategy","trades","avg_net","median_net","total_net"]
    res["win_rate"] = (df["net"] > 0).groupby(df["strategy"]).mean().values
    return res[["strategy","trades","win_rate","avg_net","median_net","total_net"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", nargs="?", default=UPBIT_PATH_DEFAULT)
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--horizon", default="24h")
    ap.add_argument("--group", default="all", choices=["all","major","alt"])
    ap.add_argument("--symbols", default="")
    ap.add_argument("--procs", type=int, default=max(1, cpu_count()-1))
    args = ap.parse_args()

    df = load_signals(args.signals)
    df = filter_group(df, args.group)
    if args.symbols:
        want = set([x.strip().upper() for x in args.symbols.split(",") if x.strip()])
        df = df[df["symbol"].isin(want)].copy()

    if args.horizon.endswith("h"):
        horizon_h = int(args.horizon[:-1])
    else:
        horizon_h = 24

    tasks = []
    for sym, rows in df.groupby("symbol"):
        tasks.append((sym, rows.sort_values("ts"), args.timeframe, horizon_h))

    with Pool(processes=args.procs) as pool:
        parts = pool.map(simulate_symbol, tasks)

    trades = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    stats = agg_stats(trades)

    os.makedirs("./logs", exist_ok=True)
    trades_path = "./logs/bt_upbit_mp_trades.csv"
    stats_path  = "./logs/bt_upbit_mp_stats.csv"
    trades.to_csv(trades_path, index=False)
    stats.to_csv(stats_path, index=False)

    print(f"[BT][UPBIT-MP] Trades: {len(trades)} → {trades_path}")
    print(f"[BT][UPBIT-MP] Stats : {len(stats)} → {stats_path}")
    if not stats.empty:
        print("\n=== Upbit-MP Backtest Summary ===")
        print(stats.sort_values(["avg_net","win_rate"], ascending=[False, False]).to_string(index=False))

if __name__ == "__main__":
    main()