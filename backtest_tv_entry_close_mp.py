# -*- coding: utf-8 -*-
"""
TV signals backtest (MP) — timezone-safe patch
- 기존 인터페이스 유지: signals_path [--timeframe 15m] [--expiry 4h,8h] [--group alt|major|all] [--procs N]
- 엔트리/로직은 기존 버전과 동일하게 유지(오직 타임존 비교/검색만 안전하게 수정)
"""

import os
import sys
import time
import math
import argparse
from multiprocessing import Pool, get_context
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pyupbit


# ========================= Helpers (Timezone-safe) =========================

def to_utc(ts) -> pd.Timestamp:
    """Return tz-aware UTC Timestamp regardless of input."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def ensure_utc_series(s: pd.Series) -> pd.Series:
    """Make a Series of timestamps tz-aware UTC."""
    if not isinstance(s.dtype, pd.DatetimeTZDtype):
        # try parse + localize
        s = pd.to_datetime(s, errors="coerce", utc=True)
    # pandas may already be tz-aware; ensure UTC
    return s.dt.tz_convert("UTC")


def next_bar_open_idx(ohlcv: pd.DataFrame, sig_ts_utc: pd.Timestamp) -> int:
    """
    Return index of the first bar strictly AFTER signal time.
    Uses tz-aware Pandas searchsorted to avoid tz-naive/aware crashes.
    """
    ts = ensure_utc_series(ohlcv["ts"])
    # Pandas-native searchsorted with tz-aware Timestamp works.
    return int(pd.DatetimeIndex(ts).searchsorted(sig_ts_utc, side="right"))


def find_index_at_or_before(ohlcv: pd.DataFrame, when_utc: pd.Timestamp) -> int:
    """Index of bar whose 'ts' <= when_utc (tz-aware safe)."""
    ts_index = pd.DatetimeIndex(ensure_utc_series(ohlcv["ts"]))
    # position of first element > when_utc, then -1
    pos = int(ts_index.searchsorted(when_utc, side="right")) - 1
    return max(0, min(pos, len(ts_index) - 1))


def floor_minutes(dt_utc: pd.Timestamp, minutes: int) -> pd.Timestamp:
    """Floor to timeframe (tz-aware safe)."""
    dt_utc = to_utc(dt_utc)
    return dt_utc.floor(f"{minutes}min")


def timeframe_minutes(tf: str) -> int:
    s = tf.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("d"):
        return int(s[:-1]) * 1440
    return 15


# ========================= Data Load =========================

def get_ohlcv_safe(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV via pyupbit and return DataFrame with tz-aware 'ts' column."""
    iv = timeframe
    df = pyupbit.get_ohlcv(ticker=symbol, interval=iv, count=500)
    if df is None or len(df) == 0:
        raise RuntimeError(f"OHLCV is None or empty for {symbol}")
    df = df.reset_index().rename(columns={"index": "ts"})
    # pyupbit index is tz-naive in localtime or UTC depending on env; force UTC tz-aware
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    else:
        # very defensive: synthesize
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        df["ts"] = pd.date_range(end=now, periods=len(df), freq="T", tz="UTC")
    # sort + dropna
    df = df.sort_values("ts").reset_index(drop=True)
    df = df[df["ts"].notna()].reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "volume"]]


# ========================= Entry/Exit Logic (기존 유지) =========================

def simulate_one_trade(
    ohlcv: pd.DataFrame,
    sig_ts: pd.Timestamp,
    side: str,
    tp: float,
    sl: float,
    expiry_h: int,
    entry_mode: str = "signal_price_or_better"
) -> Dict[str, Any]:
    """
    Long-only 가정. 기존 엔트리/조건은 유지. (오직 시간 비교만 tz-safe)
    entry_mode:
      - "close": 신호 다음 봉 시가로 진입
      - "signal_price_or_better": 신호 시점(해당 1분 가격) 이하로 체결되면 진입 (기존 로직 유지용)
    """
    # --- Normalize & timeline
    sig_ts_utc = to_utc(sig_ts)
    tf_min = 15  # 분석 OHLCV가 15m라는 가정(기존 스크립트와 동일)
    i0 = next_bar_open_idx(ohlcv, sig_ts_utc)  # next bar open
    if i0 >= len(ohlcv) - 1:
        return {}

    # signal reference price (use close of bar containing signal time at 1m granularity emulation)
    # 기존 로직 유지: 신호 직후 첫 15분봉 시가보다 유리한 가격을 허용
    price_ref = float(ohlcv["open"].iloc[i0])

    # entry price
    if entry_mode == "close":
        entry_idx = i0
        entry_price = float(ohlcv["open"].iloc[entry_idx])
    else:
        # signal_price_or_better: next bar 구간에서 price <= price_ref가 나오면 그 봉 시가 체결로 간주(간단화)
        entry_idx = i0
        entry_price = float(ohlcv["open"].iloc[entry_idx])
        # (간단화) 기존과 동일한 엔트리 조건 유지 — 필요시 더 정교화 가능
        # 만약 시가가 더 비싸면 스킵
        if entry_price > price_ref:
            return {}

    # expiry
    expiry_ts = sig_ts_utc + pd.Timedelta(hours=expiry_h)
    idx_exp = find_index_at_or_before(ohlcv, expiry_ts)
    if idx_exp <= entry_idx:
        return {}

    # simulate path: from entry_idx .. idx_exp
    path = ohlcv.iloc[entry_idx: idx_exp + 1].reset_index(drop=True)
    fee = 0.001  # round-trip 0.1%

    tp_mul = 1.0 + (tp / 100.0)
    sl_mul = 1.0 - (sl / 100.0)

    # check TP/SL with H/L sweep
    hit_tp = False
    hit_sl = False
    exit_price = float(path["close"].iloc[-1])  # default expiry at final close

    for k in range(len(path)):
        high = float(path["high"].iloc[k])
        low = float(path["low"].iloc[k])
        # Long-only
        if not hit_tp and high >= entry_price * tp_mul:
            hit_tp = True
            exit_price = entry_price * tp_mul
            break
        if not hit_sl and low <= entry_price * sl_mul:
            hit_sl = True
            exit_price = entry_price * sl_mul
            break

    gross = (exit_price - entry_price) / entry_price
    net = gross - fee

    return {
        "entry_ts": to_utc(ohlcv["ts"].iloc[entry_idx]),
        "exit_ts": to_utc(ohlcv["ts"].iloc[min(idx_exp, entry_idx + len(path) - 1)]),
        "entry": entry_price,
        "exit": exit_price,
        "tp": tp,
        "sl": sl,
        "expiry_h": expiry_h,
        "net": net,
        "gross": gross,
    }


# ========================= Task & Worker =========================

def simulate_symbol(task: Dict[str, Any]) -> pd.DataFrame:
    """
    task keys:
      symbol, rows(DataFrame of signals for symbol), timeframe, tps, sls, expiries, entry_mode
    """
    symbol = task["symbol"]
    rows = task["rows"]
    timeframe = task["timeframe"]
    tps = task["tps"]
    sls = task["sls"]
    expiries = task["expiries"]
    entry_mode = task.get("entry_mode", "signal_price_or_better")

    # load ohlcv
    try:
        ohlcv = get_ohlcv_safe(symbol, timeframe)
    except Exception as e:
        print(f"[{symbol}] OHLCV load error: {e}")
        return pd.DataFrame()

    trades = []
    for _, s in rows.iterrows():
        # ts는 이미 UTC로 파싱되어 있어야 함(아래 main에서 보장)
        sig_ts = pd.Timestamp(s["ts"])
        side = str(s.get("side", "support") or "support")

        for tp, sl in zip(tps, sls):
            for exph in expiries:
                tr = simulate_one_trade(
                    ohlcv=ohlcv,
                    sig_ts=sig_ts,
                    side=side,
                    tp=tp,
                    sl=sl,
                    expiry_h=exph,
                    entry_mode=entry_mode
                )
                if tr:
                    tr.update({
                        "symbol": symbol,
                        "signal_ts": sig_ts,
                        "side": side,
                        "tp": tp,
                        "sl": sl,
                        "strategy": f"{tp}/{sl}_{exph}h",
                    })
                    trades.append(tr)

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    # ensure tz-aware
    df["entry_ts"] = df["entry_ts"].apply(to_utc)
    df["exit_ts"] = df["exit_ts"].apply(to_utc)
    df["signal_ts"] = df["signal_ts"].apply(to_utc)
    return df


# ========================= Main =========================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="path to signals_tv.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="4h,8h", help="comma list, e.g. 4h,8h")
    ap.add_argument("--group", default="all", choices=["all", "alt", "major"])
    ap.add_argument("--procs", type=int, default=4)
    # 엔트리 모드(기존 유지). 필요시 'close'로 바꿔 테스트 가능.
    ap.add_argument("--entry-mode", default="signal_price_or_better", choices=["signal_price_or_better", "close"])
    return ap.parse_args()


def main():
    args = parse_args()

    # Load signals
    df = pd.read_csv(args.signals)
    # 표준 컬럼 가정: ts,event,side,level,touches,symbol,timeframe,extra,source,host,message
    if "ts" not in df.columns or "symbol" not in df.columns:
        print("signals file missing required columns (ts, symbol).")
        sys.exit(1)

    # tz-aware UTC 보정
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts", "symbol"]).reset_index(drop=True)

    # 그룹 필터
    majors = {"KRW-BTC", "KRW-ETH"}
    if args.group == "major":
        df = df[df["symbol"].isin(majors)]
    elif args.group == "alt":
        df = df[~df["symbol"].isin(majors)]

    if df.empty:
        print("No signals after filtering.")
        return

    # 전략 파라미터(기존 유지)
    STRATS = [
        (1.5, 1.0),
        (2.0, 1.25),
        (1.0, 0.75),
        (1.25, 1.0),
        (1.75, 1.25),
        (0.8, 0.8),
    ]
    tps = [tp for tp, _ in STRATS]
    sls = [sl for _, sl in STRATS]

    # 만기 시간 파싱
    expiries_h = []
    for x in (args.expiry.split(",") if args.expiry else []):
        x = x.strip().lower()
        if x.endswith("h"):
            expiries_h.append(int(x[:-1]))
    if not expiries_h:
        expiries_h = [4, 8]

    # 심볼별 그룹화
    tasks = []
    for sym, g in df.groupby("symbol"):
        tasks.append({
            "symbol": sym,
            "rows": g.copy(),
            "timeframe": args.timeframe,
            "tps": tps,
            "sls": sls,
            "expiries": expiries_h,
            "entry_mode": args.entry_mode,
        })

    print(f"[BT] starting with {len(tasks)} tasks using {args.procs} procs...")

    if not tasks:
        print("No trades generated.")
        return

    # Windows 안전 실행
    ctx = get_context("spawn")
    with ctx.Pool(processes=max(1, args.procs)) as pool:
        parts = pool.map(simulate_symbol, tasks)

    trades = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if trades.empty:
        print("No trades generated.")
        return

    # 결과 저장
    os.makedirs("./logs", exist_ok=True)
    out_trades = "./logs/bt_tv_entryclose_trades.csv"
    out_stats = "./logs/bt_tv_entryclose_stats.csv"

    # 집계
    def agg_stats(g: pd.DataFrame) -> pd.Series:
        wins = (g["net"] > 0).sum()
        return pd.Series({
            "trades": len(g),
            "win_rate": wins / max(1, len(g)),
            "avg_net": g["net"].mean(),
            "median_net": g["net"].median(),
            "total_net": g["net"].sum(),
        })

    trades.to_csv(out_trades, index=False)

    # 전략별 집계(만기/TP/SL 묶어서 보기)
    trades["strategy"] = trades["strategy"].astype(str)
    stats = trades.groupby("strategy", as_index=False, dropna=False).apply(agg_stats)
    stats.to_csv(out_stats, index=False)

    print(f"Trades saved: {out_trades} (rows={len(trades)})")
    print(f"Stats  saved: {out_stats} (rows={len(stats)})")


if __name__ == "__main__":
    main()