# -*- coding: utf-8 -*-
"""
backtest_upbit_multi_strategies.py
- 입력: ./logs/signals_upbit.csv (기본)
- 출력: ./logs/bt_upbit_results.csv, ./logs/bt_upbit_trades.csv
- 시나리오(5): (TV용과 동일)
"""

import os, sys, time
from datetime import timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import pyupbit

INPUT_PATH_DEFAULT = "./logs/signals_upbit.csv"
OUT_RESULTS = "./logs/bt_upbit_results.csv"
OUT_TRADES  = "./logs/bt_upbit_trades.csv"
CACHE_DIR   = "./cache_ohlcv"

TIMEFRAME   = "15m"
FEE_ENTRY   = 0.0005
FEE_EXIT    = 0.0005
EXPIRY_H    = 24

STRATEGIES = [
    ("stable_1p5_1p0",      0.015,  -0.010),
    ("aggressive_2p0_1p25", 0.020,  -0.0125),
    ("scalp_1p0_0p75",      0.010,  -0.0075),
    ("tp1p25_sl1p0",        0.0125, -0.010),
    ("tp1p0_sl1p0",         0.010,  -0.010),
    ("bold_0p8_0p8",        0.008,  -0.008),  # ★ 신규 시나리오 6
]


os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("./logs", exist_ok=True)

def read_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise RuntimeError("signals 파일에 'ts' 컬럼이 필요합니다.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce", format="ISO8601")
    df = df.dropna(subset=["ts"]).copy()
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    else:
        raise RuntimeError("signals 파일에 'symbol' 컬럼이 필요합니다.")
    if "event" in df.columns:
        df["event"] = df["event"].astype(str).str.lower().str.strip()
    else:
        df["event"] = "unknown"
    if "timeframe" not in df.columns or df["timeframe"].isna().all():
        df["timeframe"] = TIMEFRAME
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def cache_path(symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "_")
    return os.path.join(CACHE_DIR, f"ohlcv_{safe}_{timeframe}.csv")

def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    p = cache_path(symbol, timeframe)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce", format="ISO8601")
            df = df.dropna(subset=["ts"]).reset_index(drop=True)
            return df
        except Exception:
            pass
    raw = pyupbit.get_ohlcv(ticker=symbol, interval=timeframe, count=2000)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"OHLCV 로드 실패: {symbol} {timeframe}")
    raw = raw.copy()
    raw["ts"] = pd.to_datetime(raw.index, utc=True)
    raw = raw.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume","value":"value"})
    raw = raw[["ts","open","high","low","close","volume","value"]].reset_index(drop=True)
    raw.to_csv(p, index=False)
    return raw

def next_bar_open_idx(ohlcv: pd.DataFrame, signal_ts: pd.Timestamp) -> int:
    ts = ohlcv["ts"].to_numpy()
    idx = np.searchsorted(ts, np.datetime64(signal_ts), side="right")
    return int(idx)

def expiry_close_idx(ohlcv: pd.DataFrame, entry_ts: pd.Timestamp, hours: int) -> int:
    target = entry_ts + pd.Timedelta(hours=hours)
    ts = ohlcv["ts"].to_numpy()
    j = int(np.searchsorted(ts, np.datetime64(target), side="right"))
    j = max(0, min(j, len(ohlcv)-1))
    return j

def simulate(symbol: str, sig_ts_list: List[pd.Timestamp], tp: float, sl: float) -> Tuple[List[Dict], Dict]:
    ohlcv = load_ohlcv(symbol, TIMEFRAME)
    trades = []
    open_until = None
    for sig_ts in sig_ts_list:
        if open_until is not None and sig_ts < open_until:
            continue
        i = next_bar_open_idx(ohlcv, sig_ts)
        if i >= len(ohlcv):
            continue
        entry_time = ohlcv["ts"].iloc[i]
        entry_price = float(ohlcv["open"].iloc[i])
        entry_price_fee = entry_price * (1 + FEE_ENTRY)

        j_end = expiry_close_idx(ohlcv, entry_time, EXPIRY_H)
        exit_idx = None
        exit_reason = None
        for k in range(i, j_end+1):
            hi = float(ohlcv["high"].iloc[k]); lo = float(ohlcv["low"].iloc[k])
            if hi >= entry_price_fee * (1 + tp):
                exit_idx = k; exit_reason = "TP"; break
            if lo <= entry_price_fee * (1 + sl):
                exit_idx = k; exit_reason = "SL"; break
        if exit_idx is None:
            exit_idx = j_end; exit_reason = "EXP"

        exit_time = ohlcv["ts"].iloc[exit_idx]
        raw_exit = float(ohlcv["close"].iloc[exit_idx])
        exit_price_fee = raw_exit * (1 - FEE_EXIT)
        ret = (exit_price_fee - entry_price_fee) / entry_price_fee

        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_price": entry_price_fee,
            "exit_time": exit_time,
            "exit_price": exit_price_fee,
            "reason": exit_reason,
            "ret": ret
        })
        open_until = exit_time

    if trades:
        r = pd.DataFrame(trades)["ret"]
        stats = dict(
            n=len(trades),
            win=int((r>0).sum()),
            winrate=float((r>0).mean()),
            avg=float(r.mean()),
            med=float(r.median()),
            std=float(r.std(ddof=1)) if len(r)>1 else 0.0,
            sum=float(r.sum()),
            max=float(r.max()),
            min=float(r.min()),
        )
    else:
        stats = dict(n=0, win=0, winrate=0.0, avg=0.0, med=0.0, std=0.0, sum=0.0, max=0.0, min=0.0)
    return trades, stats

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT_PATH_DEFAULT
    sig = read_signals(path)

    # Upbit 쪽은 기본적으로 event가 box_breakout/line_breakout 중심이므로 그대로 사용
    sig = sig[sig["symbol"].str.startswith("KRW-")].copy()

    symbols = sorted(sig["symbol"].unique().tolist())
    rows_results = []
    rows_trades  = []

    t0 = time.time()
    for name, tp, sl in STRATEGIES:
        processed = 0
        for sym in symbols:
            st = sig[sig["symbol"]==sym]["ts"].tolist()
            trades, stats = simulate(sym, st, tp, sl)
            for tr in trades:
                rows_trades.append(dict(strategy=name, **tr))
            rows_results.append({
                "strategy": name, "symbol": sym, **stats
            })
            processed += len(st)
        print(f"[BT-UPBIT] {name} done. processed_signals={processed}")
    pd.DataFrame(rows_results).to_csv(OUT_RESULTS, index=False)
    pd.DataFrame(rows_trades).to_csv(OUT_TRADES, index=False)
    print(f"[BT-UPBIT] ✅ 완료: results -> {OUT_RESULTS} , trades -> {OUT_TRADES} (took {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
