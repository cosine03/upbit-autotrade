# backtest_tv_multi_strategies.py
import os
import argparse
from functools import lru_cache
from math import ceil

import numpy as np
import pandas as pd
from sr_engine.data import get_ohlcv

# ------------------------ 유틸 ------------------------
def ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")

def ensure_ts_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            df["ts"] = idx
        else:
            for c in ("timestamp", "time", "datetime", "date", "ts"):
                if c in df.columns:
                    df["ts"] = pd.to_datetime(df[c], utc=True, errors="coerce")
                    break
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def tf_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 60 * 24
    return 15

@lru_cache(maxsize=None)
def get_ohlcv_cached(symbol: str, timeframe: str) -> pd.DataFrame:
    df = get_ohlcv(symbol, timeframe)
    df = ensure_ts_column(df)
    for c in ("open","high","low","close"):
        if c not in df.columns:
            raise RuntimeError(f"OHLCV에 '{c}'가 없습니다: {symbol} {timeframe}")
    return df

def next_bar_open_idx(ohlcv: pd.DataFrame, signal_ts: pd.Timestamp) -> int | None:
    ts = ohlcv["ts"].to_numpy(dtype="datetime64[ns]")
    sig = np.datetime64(to_utc(signal_ts).to_datetime64())
    idx = ts.searchsorted(sig, side="right")
    return int(idx) if idx < len(ts) else None

def bars_for_24h(timeframe: str) -> int:
    return max(1, ceil(24 * 60 / tf_to_minutes(timeframe)))

# ------------------------ 시그널 로드 ------------------------
def load_tv_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"ts","event","side","symbol"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"필수 컬럼 누락: {miss}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","symbol","side"]).reset_index(drop=True)

    df = df[df["side"].str.lower() == "support"].copy()
    df["event"] = df["event"].str.lower()
    tv_events = {"level2_detected","price_in_box"}
    df = df[df["event"].isin(tv_events)].copy()

    if "timeframe" not in df.columns or df["timeframe"].isna().all():
        df["timeframe"] = "15m"
    return df[["ts","event","side","symbol","timeframe"]].sort_values("ts").reset_index(drop=True)

# ------------------------ 시뮬레이션 ------------------------
def simulate_symbol(
    sym: str,
    df_sig: pd.DataFrame,
    timeframe: str,
    tp: float,
    sl: float,
    fee_one_side: float = 0.0005,
) -> list[dict]:
    ohlcv = get_ohlcv_cached(sym, timeframe)
    trades = []
    max_bars = bars_for_24h(timeframe)

    ts_arr = ohlcv["ts"].to_numpy()
    open_arr  = ohlcv["open"].to_numpy(dtype=float)
    high_arr  = ohlcv["high"].to_numpy(dtype=float)
    low_arr   = ohlcv["low"].to_numpy(dtype=float)
    close_arr = ohlcv["close"].to_numpy(dtype=float)

    for _, s in df_sig.iterrows():
        sig_ts = to_utc(s["ts"])
        i = next_bar_open_idx(ohlcv, sig_ts)
        if i is None or i >= len(ohlcv): 
            continue

        entry_ts = to_utc(ts_arr[i])
        entry_px = float(open_arr[i])
        target_px = entry_px * (1 + tp)
        stop_px   = entry_px * (1 - sl)

        exit_idx = None
        exit_px = None
        exit_reason = None

        last_idx = min(len(ohlcv)-1, i + max_bars - 1)
        for j in range(i, last_idx + 1):
            hi = float(high_arr[j]); lo = float(low_arr[j])
            hit_tp = hi >= target_px
            hit_sl = lo <= stop_px
            if hit_tp and hit_sl:
                exit_idx, exit_px, exit_reason = j, stop_px, "SL_both_hit"
                break
            if hit_sl:
                exit_idx, exit_px, exit_reason = j, stop_px, "SL"
                break
            if hit_tp:
                exit_idx, exit_px, exit_reason = j, target_px, "TP"
                break

        if exit_idx is None:
            exit_idx = last_idx
            exit_px = float(close_arr[exit_idx])
            exit_reason = "EXPIRY"

        gross = (exit_px / entry_px) - 1.0
        net = gross - (fee_one_side + fee_one_side)

        trades.append({
            "symbol": sym,
            "signal_ts": sig_ts.isoformat(),
            "entry_ts": to_utc(ts_arr[i]).isoformat(),
            "entry_px": entry_px,
            "exit_ts": to_utc(ts_arr[exit_idx]).isoformat(),
            "exit_px": exit_px,
            "exit_reason": exit_reason,
            "tp": tp, "sl": sl,
            "ret_gross": gross,
            "ret_net": net,
        })
    return trades

def simulate_all(df_sig: pd.DataFrame, timeframe: str, strategies: list[tuple[str,float,float]]):
    out = []
    symbols = sorted(df_sig["symbol"].unique().tolist())
    for name, tp, sl in strategies:
        for sym in symbols:
            rows = df_sig[df_sig["symbol"] == sym]
            tr = simulate_symbol(sym, rows, timeframe, tp, sl)
            for t in tr: t["strategy"] = name
            out.extend(tr)
        print(f"[BT] {name} done. sym={len(symbols)} trades={sum(1 for t in out if t['strategy']==name)}")

    trades_df = pd.DataFrame(out)
    if trades_df.empty:
        stats_df = pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"])
        return trades_df, stats_df

    def _agg(g):
        trades = len(g)
        wins = int((g["ret_net"] > 0).sum())
        win_rate = wins / trades if trades else 0.0
        return pd.Series({
            "trades": trades,
            "win_rate": round(win_rate, 4),
            "avg_net": round(float(g["ret_net"].mean()), 6),
            "median_net": round(float(g["ret_net"].median()), 6),
            "total_net": round(float(g["ret_net"].sum()), 6),
        })
    stats_df = trades_df.groupby("strategy", as_index=False).apply(_agg, include_groups=False)
    stats_df = stats_df.sort_values(["total_net","win_rate"], ascending=[False, False])
    return trades_df, stats_df

# ------------------------ 메인 ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", nargs="?", default="./logs/signals_tv.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--out-trades", default="./logs/bt_tv_trades.csv")
    ap.add_argument("--out-stats",  default="./logs/bt_tv_stats.csv")
    args = ap.parse_args()

    ensure_dir(args.out_trades); ensure_dir(args.out_stats)

    df_sig = load_tv_signals(args.signals)
    df_sig["timeframe"] = args.timeframe

    strategies = [
        ("stable_1.5/1.0",     0.015,  0.010),
        ("aggressive_2.0/1.25",0.020,  0.0125),
        ("scalp_1.0/0.75",     0.010,  0.0075),
        ("mid_1.25/1.0",       0.0125, 0.010),
        ("mid_1.75/1.25",      0.0175, 0.0125),
        ("tight_0.8/0.8",      0.008,  0.008),
    ]

    trades_df, stats_df = simulate_all(df_sig, args.timeframe, strategies)
    trades_df.to_csv(args.out_trades, index=False)
    stats_df.to_csv(args.out_stats, index=False)

    print("\n=== TV Backtest (Long-only / 24h expiry / 0.1% fees) ===")
    if trades_df.empty:
        print("No trades produced.")
        return
    print(f"Trades saved: {args.out_trades} (rows={len(trades_df)})")
    print(f"Stats  saved: {args.out_stats} (rows={len(stats_df)})\n")
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    main()
