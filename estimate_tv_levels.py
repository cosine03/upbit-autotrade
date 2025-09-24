#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TV 로그(enriched) 보강: 이벤트 시점의 레벨 박스/라인 근사치 추정
- 입력: signals_tv.csv (원본)
- 출력: signals_tv_enriched.csv (distance_pct 포함)
- sr_robust.get_ohlcv 사용(가능 시) + 로컬 CSV fallback
- UTC/타임존 안전화
"""

from __future__ import annotations
import argparse
import os
from typing import Optional
import numpy as np
import pandas as pd

try:
    from sr_robust import get_ohlcv  # type: ignore
except Exception:
    get_ohlcv = None

def to_utc_series(obj) -> pd.Series:
    return pd.to_datetime(obj, utc=True, errors="coerce")

def load_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    df = None
    if callable(get_ohlcv):
        try:
            df = get_ohlcv(symbol, timeframe)
        except Exception:
            df = None
    if df is None:
        path = os.path.join("data", "ohlcv", f"{symbol}_{timeframe}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None
    if df is None or df.empty:
        return None
    need = {"ts", "open", "high", "low", "close"}
    if not need.issubset(set(map(str, df.columns))):
        return None
    df = df.copy()
    df["ts"] = to_utc_series(df["ts"])
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close"]]

def nearest_box_proxy(ohlcv: pd.DataFrame, i_sig: int, lookback: int = 96) -> Optional[tuple]:
    """
    시그널 이전 N봉에서 박스(저항/지지) 근사: 하위/상위 분위값으로 박스 대략 추정.
    """
    i0 = max(0, i_sig - lookback)
    seg = ohlcv.iloc[i0:i_sig]
    if seg.empty:
        return None
    # 박스 하단/상단 근사(분위값)
    low_q = np.quantile(seg["low"].to_numpy(), 0.8)   # 저항 박스라면 상단 쪽
    high_q = np.quantile(seg["high"].to_numpy(), 0.9)
    return (float(low_q), float(high_q))

def enrich(signals_csv: str, timeframe: str, out_csv: str):
    df = pd.read_csv(signals_csv)
    # 필수 컬럼
    need = ["ts", "event", "symbol"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"signals '{c}' missing")
    df["ts"] = to_utc_series(df["ts"])
    df = df.dropna(subset=["ts", "event", "symbol"]).reset_index(drop=True)

    rows = []
    g = df.groupby("symbol")
    for sym, sub in g:
        ohlcv = load_ohlcv(sym, timeframe)
        if ohlcv is None or ohlcv.empty:
            continue
        ts_ns = to_utc_series(ohlcv["ts"]).view("int64").to_numpy()
        closes = ohlcv["close"].to_numpy(float)

        for r in sub.itertuples(index=False):
            sig_ts = getattr(r, "ts")
            if pd.isna(sig_ts):
                continue
            sig_ns = to_utc_series([sig_ts]).view("int64")[0]
            i_sig = int(np.searchsorted(ts_ns, sig_ns, side="right")) - 1
            if i_sig < 1:
                continue
            entry_i = i_sig + 1 if (i_sig + 1) < len(ohlcv) else i_sig
            entry_px = float(closes[entry_i])

            # 박스 근사
            box = nearest_box_proxy(ohlcv, i_sig)
            if box is None:
                continue
            b_lo, b_hi = box
            # 박스 중앙 기준 괴리 (%)
            mid = 0.5 * (b_lo + b_hi)
            dist = abs(entry_px - mid) / mid * 100.0 if mid > 0 else np.nan

            rows.append({
                **r._asdict(),
                "entry_price": entry_px,
                "box_lo_est": b_lo,
                "box_hi_est": b_hi,
                "box_mid_est": mid,
                "distance_pct": dist,
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[EST] done. saved -> {out_csv}")
    print(f"[EST] success {len(out)}/{len(df)} rows")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv")
    ap.add_argument("--http-timeout", type=float, default=8.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--throttle", type=float, default=0.10)
    ap.add_argument("--dist-max", type=float, default=None, help="e.g., 0.02 for 0.02%")
    args = ap.parse_args()
    enrich(args.signals, args.timeframe, args.out)

if __name__ == "__main__":
    main()