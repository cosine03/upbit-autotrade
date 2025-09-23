# -*- coding: utf-8 -*-
"""
backtest_tv_breakout_close_sp.py

- TradingView 기반 breakout 신호 (signals_tv.csv) 백테스트
- 엔트리 방식: breakout 후 "close 기준"
- 조건:
    * touches >= N (예: 2, 3)
    * 롱 온리
    * TP / SL %
    * 만료시간 (24h 기본)
    * 수수료 왕복 0.1%

실행 예시:
    python backtest_tv_breakout_close_sp.py ./logs/signals_tv.csv \
        --timeframe 15m --expiry 24h --tp 1.5 --sl 1.0 --touches 2 --touches-alt 3
"""

import os, argparse
import numpy as np
import pandas as pd
import pyupbit
from datetime import timedelta

# -----------------------
# Utils
# -----------------------
def parse_expiry(exp: str) -> float:
    """지원: Xm, Xh, Xd"""
    exp = exp.strip().lower()
    if exp.endswith("m"):
        return int(exp[:-1]) / 60.0
    if exp.endswith("h"):
        return int(exp[:-1])
    if exp.endswith("d"):
        return int(exp[:-1]) * 24.0
    return float(exp)

def get_ohlcv(symbol: str, tf: str = "15m", count: int = 400) -> pd.DataFrame:
    try:
        df = pyupbit.get_ohlcv(symbol, interval=tf, count=count)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df["ts"] = pd.to_datetime(df["index"], utc=True)
        return df[["ts","open","high","low","close","volume"]].copy()
    except Exception as e:
        print(f"[{symbol}] OHLCV load error: {e}")
        return None

def idx_of_bar(ts: np.ndarray, key: pd.Timestamp) -> int:
    """key 시각이 속하는 bar index 찾기"""
    if not isinstance(key, pd.Timestamp):
        key = pd.Timestamp(key)
    if key.tz is None:
        key = key.tz_localize("UTC")
    else:
        key = key.tz_convert("UTC")
    key64 = key.to_datetime64()
    idx = int(np.searchsorted(ts, key64, side="right")) - 1
    return max(0, idx)

# -----------------------
# Core simulation
# -----------------------
def simulate_symbol_breakout_close(symbol, ohlcv, signals, tp_pct, sl_pct, fee_rt, timeframe, expiry_hours, touches_req=2):
    trades = []
    ts = ohlcv["ts"].to_numpy()

    for _, s in signals.iterrows():
        sig_ts = pd.Timestamp(s["ts"])
        if sig_ts.tz is None:
            sig_ts = sig_ts.tz_localize("UTC")
        else:
            sig_ts = sig_ts.tz_convert("UTC")

        # 최소 터치 조건
        t = int(s.get("touches", 0))
        if t < touches_req:
            continue

        # breakout 엔트리: 신호 시점 bar close 가격
        i_sig = idx_of_bar(ts, sig_ts)
        if i_sig < 0 or i_sig >= len(ohlcv)-1:
            continue
        entry_px = float(ohlcv["close"].iloc[i_sig])

        # TP / SL 레벨
        tp_px = entry_px * (1 + tp_pct/100.0)
        sl_px = entry_px * (1 - sl_pct/100.0)

        # 만료 bar
        expiry_ts = sig_ts + timedelta(hours=expiry_hours)
        ts_exp = expiry_ts.to_datetime64()
        idx_exp = int(np.searchsorted(ts, ts_exp, side="left"))
        if idx_exp <= i_sig:
            continue
        df_future = ohlcv.iloc[i_sig+1:idx_exp+1]

        exit_px, exit_reason = None, None
        for _, row in df_future.iterrows():
            lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])
            if lo <= sl_px:
                exit_px, exit_reason = sl_px, "SL"
                break
            if hi >= tp_px:
                exit_px, exit_reason = tp_px, "TP"
                break
        if exit_px is None:
            exit_px, exit_reason = float(df_future["close"].iloc[-1]), "EXP"

        net = (exit_px - entry_px) / entry_px
        net -= fee_rt

        trades.append({
            "symbol": symbol,
            "sig_ts": sig_ts.isoformat(),
            "entry_px": entry_px,
            "exit_px": exit_px,
            "exit_reason": exit_reason,
            "tp": tp_pct,
            "sl": sl_pct,
            "expiry_h": expiry_hours,
            "touches_req": touches_req,
            "touches": t,
            "net": net
        })
    return pd.DataFrame(trades)

# -----------------------
# Aggregation
# -----------------------
def agg_stats(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "trades": len(g),
        "win_rate": np.mean(g["net"] > 0) if len(g) else np.nan,
        "avg_net": g["net"].mean() if len(g) else np.nan,
        "median_net": g["net"].median() if len(g) else np.nan,
        "total_net": g["net"].sum() if len(g) else np.nan,
    })

def summarize(trades: pd.DataFrame, label: str):
    if trades is None or trades.empty:
        print(f"(no trades for {label})")
        return pd.DataFrame()
    stats = trades.groupby("strategy", as_index=False).apply(agg_stats)
    print(f"\n=== {label} ===")
    print(stats.to_string(index=False))
    return stats

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="24h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)  # 0.1%
    ap.add_argument("--touches", type=int, default=2)
    ap.add_argument("--touches-alt", type=int, default=3)
    args = ap.parse_args()

    expiry_hours = parse_expiry(args.expiry)

    df_sig = pd.read_csv(args.signals)
    if "ts" not in df_sig.columns:
        raise ValueError("signals CSV must have 'ts'")
    df_sig["ts"] = pd.to_datetime(df_sig["ts"], utc=True)

    symbols = sorted(df_sig["symbol"].dropna().unique())
    print(f"[BT] symbols={len(symbols)} signals(rows)={len(df_sig)} timeframe={args.timeframe}")

    trades_all = []
    for sym in symbols:
        ohlcv = get_ohlcv(sym, args.timeframe, count=500)
        if ohlcv is None: 
            continue
        rows = df_sig[df_sig["symbol"]==sym]
        tr = simulate_symbol_breakout_close(
            sym, ohlcv, rows,
            tp_pct=args.tp, sl_pct=args.sl,
            fee_rt=args.fee,
            timeframe=args.timeframe,
            expiry_hours=expiry_hours,
            touches_req=args.touches
        )
        if not tr.empty:
            tr["strategy"] = f"touch{args.touches}_{args.tp}/{args.sl}_{args.expiry}"
            trades_all.append(tr)

    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    stats = summarize(trades, f"touches>={args.touches}")

    # 추가 비교 (touches_alt)
    if args.touches_alt:
        trades_all_alt = []
        for sym in symbols:
            ohlcv = get_ohlcv(sym, args.timeframe, count=500)
            if ohlcv is None: 
                continue
            rows = df_sig[df_sig["symbol"]==sym]
            tr = simulate_symbol_breakout_close(
                sym, ohlcv, rows,
                tp_pct=args.tp, sl_pct=args.sl,
                fee_rt=args.fee,
                timeframe=args.timeframe,
                expiry_hours=expiry_hours,
                touches_req=args.touches_alt
            )
            if not tr.empty:
                tr["strategy"] = f"touch{args.touches_alt}_{args.tp}/{args.sl}_{args.expiry}"
                trades_all_alt.append(tr)
        trades_alt = pd.concat(trades_all_alt, ignore_index=True) if trades_all_alt else pd.DataFrame()
        stats_alt = summarize(trades_alt, f"touches>={args.touches_alt}")
    else:
        trades_alt, stats_alt = pd.DataFrame(), pd.DataFrame()

    os.makedirs("logs", exist_ok=True)
    trades.to_csv("./logs/bt_tv_breakout_trades.csv", index=False)
    stats.to_csv("./logs/bt_tv_breakout_stats.csv", index=False)
    trades_alt.to_csv("./logs/bt_tv_breakout_trades_alt.csv", index=False)
    stats_alt.to_csv("./logs/bt_tv_breakout_stats_alt.csv", index=False)
    print("[BT] 저장 완료.")

if __name__ == "__main__":
    main()