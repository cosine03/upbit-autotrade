# -*- coding: utf-8 -*-
"""
backtest_tv_breakout_close_sp.py

단일 프로세스 버전 (debug 용)
- TradingView signals (signals_tv.csv) 기반
- breakout 엔트리: "touches >= N" 후 다음 종가가 레벨 위(저항) 또는 아래(지지)
- long only
- expiry 시간 후 종가 청산
- TP/SL 지원
"""

import os
import argparse
import pandas as pd
import numpy as np
import pyupbit
from datetime import timedelta

# ====================== 데이터 유틸 ======================

def get_ohlcv(symbol: str, timeframe: str = "15m", count: int = 5000) -> pd.DataFrame:
    """Upbit OHLCV 불러오기"""
    iv_map = {"1m": "minute1", "3m": "minute3", "5m": "minute5",
              "15m": "minute15", "30m": "minute30", "60m": "minute60",
              "240m": "minute240", "1d": "day"}
    iv = iv_map.get(timeframe, "minute15")
    df = pyupbit.get_ohlcv(symbol, interval=iv, count=count)
    if df is None:
        return None
    df = df.reset_index()
    df.rename(columns={"index": "ts"}, inplace=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)  # tz-aware UTC
    return df

def idx_of_bar(ts: np.ndarray, when_utc: pd.Timestamp) -> int:
    """
    ts: np.ndarray of pandas Timestamps
    when_utc: tz-aware UTC Timestamp
    """
    ts64 = ts.astype("datetime64[ns]")  # TZ-naive 변환
    if when_utc.tzinfo:
        key64 = when_utc.tz_convert("UTC").to_datetime64()
    else:
        key64 = when_utc.tz_localize("UTC").to_datetime64()

    idx = int(np.searchsorted(ts64, key64, side="right")) - 1
    if idx < 0 or idx >= len(ts64):
        return -1
    return idx

# ====================== 시뮬레이션 ======================

def simulate_symbol_breakout_close(symbol: str, ohlcv: pd.DataFrame, signals: pd.DataFrame,
                                   tp_pct: float, sl_pct: float, fee_rt: float,
                                   timeframe: str, expiry_hours: float,
                                   min_touches: int = 2) -> pd.DataFrame:
    """
    breakout 엔트리: touches >= min_touches 이후 다음 종가가 레벨 돌파 시 진입
    """

    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    # ts 배열 준비
    ts_pd = ohlcv["ts"]
    if isinstance(ts_pd.dtype, pd.DatetimeTZDtype):
        ts_pd = ts_pd.dt.tz_convert("UTC")
    else:
        ts_pd = pd.to_datetime(ts_pd, utc=True)
    ts = ts_pd.to_numpy()

    trades = []

    for _, s in signals.iterrows():
        # === 신호 시각 UTC 정규화 ===
        sig_ts = pd.Timestamp(s["ts"])
        sig_ts = sig_ts.tz_convert("UTC") if sig_ts.tzinfo else sig_ts.tz_localize("UTC")

        touches = int(s.get("touches", 0))
        if touches < min_touches:
            continue

        i_sig = idx_of_bar(ts, sig_ts)
        if i_sig < 0 or i_sig >= len(ohlcv) - 1:
            continue

        # breakout 조건: 다음 종가가 레벨 위 (resistance)
        level = float(s.get("level", 0))
        side = s.get("side", "resistance")
        close_next = float(ohlcv["close"].iloc[i_sig + 1])

        if side == "resistance":
            cond = close_next > level
        else:
            cond = close_next < level
        if not cond:
            continue

        # === 진입 ===
        entry_ts = ohlcv["ts"].iloc[i_sig + 1]
        entry_px = close_next

        # === 만기 ===
        expiry_ts = entry_ts + pd.Timedelta(hours=expiry_hours)
        expiry_idx = np.searchsorted(ts.astype("datetime64[ns]"),
                                     expiry_ts.to_datetime64(),
                                     side="left")
        if expiry_idx >= len(ohlcv):
            expiry_idx = len(ohlcv) - 1

        # === 시뮬레이션 (TP/SL) ===
        pxs = ohlcv.loc[i_sig+2:expiry_idx, ["ts", "high", "low", "close"]]
        exit_px = pxs["close"].iloc[-1]
        exit_ts = pxs["ts"].iloc[-1]

        hit_tp, hit_sl = False, False
        for _, row in pxs.iterrows():
            if side == "resistance":
                if row["high"] >= entry_px * (1 + tp_pct/100):
                    exit_px = entry_px * (1 + tp_pct/100)
                    exit_ts = row["ts"]
                    hit_tp = True
                    break
                if row["low"] <= entry_px * (1 - sl_pct/100):
                    exit_px = entry_px * (1 - sl_pct/100)
                    exit_ts = row["ts"]
                    hit_sl = True
                    break
            else:
                # support breakout (long only라서 실제론 잘 안 씀)
                if row["low"] <= entry_px * (1 - tp_pct/100):
                    exit_px = entry_px * (1 - tp_pct/100)
                    exit_ts = row["ts"]
                    hit_tp = True
                    break
                if row["high"] >= entry_px * (1 + sl_pct/100):
                    exit_px = entry_px * (1 + sl_pct/100)
                    exit_ts = row["ts"]
                    hit_sl = True
                    break

        net = (exit_px / entry_px - 1) - fee_rt
        trades.append({
            "symbol": symbol,
            "signal_ts": sig_ts,
            "entry_ts": entry_ts,
            "entry_px": entry_px,
            "exit_ts": exit_ts,
            "exit_px": exit_px,
            "tp": tp_pct, "sl": sl_pct,
            "expiry_h": expiry_hours,
            "touches": touches,
            "side": side,
            "level": level,
            "net": net,
            "hit_tp": hit_tp,
            "hit_sl": hit_sl
        })

    return pd.DataFrame(trades)

# ====================== 메인 ======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv 경로")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="24h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)  # 왕복 0.1%
    ap.add_argument("--touches", type=int, default=2)
    args = ap.parse_args()

    expiry_hours = float(args.expiry.replace("h",""))

    df = pd.read_csv(args.signals)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    symbols = df["symbol"].dropna().unique().tolist()
    print(f"[BT] symbols={len(symbols)} signals(rows)={len(df)} timeframe={args.timeframe}")
    print(f"[BT] scenario: touches>={args.touches}, TP={args.tp} SL={args.sl}, expiry={expiry_hours}h")

    all_trades = []
    for sym in symbols:
        ohlcv = get_ohlcv(sym, args.timeframe)
        rows = df[df["symbol"] == sym]
        tr = simulate_symbol_breakout_close(sym, ohlcv, rows,
                                            tp_pct=args.tp, sl_pct=args.sl,
                                            fee_rt=args.fee,
                                            timeframe=args.timeframe,
                                            expiry_hours=expiry_hours,
                                            min_touches=args.touches)
        all_trades.append(tr)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    out_trades = "./logs/bt_tv_breakoutclose_trades.csv"
    trades.to_csv(out_trades, index=False)
    print(f"[BT] 완료. trades={len(trades)} → {out_trades}")

if __name__ == "__main__":
    main()