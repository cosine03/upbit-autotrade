# -*- coding: utf-8 -*-
"""
backtest_tv_multi_strategies.py
- 입력: ./logs/signals_tv.csv (기본)
- 출력: ./logs/bt_tv_results.csv, ./logs/bt_tv_trades.csv
- 시나리오(5):
  1) stable:   TP 1.5% / SL -1.0% / 24H
  2) aggressive: TP 2.0% / SL -1.25% / 24H
  3) scalp:    TP 1.0% / SL -0.75% / 24H
  4) tp1.25_sl1.0:  TP 1.25% / SL -1.0% / 24H
  5) tp1.0_sl1.0:   TP 1.0% / SL -1.0% / 24H
"""

import os, sys, time
from datetime import timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# pyupbit는 사전에 설치되어 있어야 합니다.
# pip install pyupbit python-dotenv
import pyupbit

from sr_engine.data import get_ohlcv

INPUT_PATH_DEFAULT = "./logs/signals_tv.csv"
OUT_RESULTS = "./logs/bt_tv_results.csv"
OUT_TRADES  = "./logs/bt_tv_trades.csv"
CACHE_DIR   = "./cache_ohlcv"

TIMEFRAME   = "15m"
FEE_ENTRY   = 0.0005  # 0.05%
FEE_EXIT    = 0.0005  # 0.05%
EXPIRY_H    = 24      # 24 hours

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
    # 표준화
    cols = {c.lower().strip(): c for c in df.columns}
    def col(name): return cols.get(name, name)
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
    """CSV 캐시 사용. 없으면 pyupbit에서 로드."""
    p = cache_path(symbol, timeframe)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce", format="ISO8601")
            df = df.dropna(subset=["ts"]).reset_index(drop=True)
            return df
        except Exception:
            pass
    # 없거나 읽기 실패 → 새로 로드
    # pyupbit.get_ohlcv: index가 datetime, tz-naive. UTC로 가정하고 tz-localize
    raw = pyupbit.get_ohlcv(ticker=symbol, interval=timeframe, count=2000)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"OHLCV 로드 실패: {symbol} {timeframe}")
    raw = raw.copy()
    raw["ts"] = pd.to_datetime(raw.index, utc=True)
    raw = raw.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume","value":"value"})
    raw = raw[["ts","open","high","low","close","volume","value"]].reset_index(drop=True)
    raw.to_csv(p, index=False)
    return raw

def ensure_utc_index(df):
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
    return df

ohlcv = get_ohlcv(sym, timeframe)
ohlcv = ensure_utc_index(ohlcv)

def next_bar_open_idx(ohlcv: pd.DataFrame, signal_ts) -> int:
    # 1) OHLCV 인덱스는 반드시 UTC
    ts = ohlcv.index
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    # 2) 시그널 시각도 UTC로 통일
    sig_ts = pd.to_datetime(signal_ts, utc=True)

    # 3) 바로 '판다스 인덱스'의 searchsorted 사용 (numpy 말고!)
    #    side="right" → 시그널이 뜬 '다음 봉 오픈'을 진입 시점으로
    idx = ts.searchsorted(sig_ts, side="right")

    # 범위 방어
    if idx >= len(ts):
        idx = len(ts) - 1
    return int(idx)


def expiry_close_idx(ohlcv: pd.DataFrame, entry_ts: pd.Timestamp, hours: int) -> int:
    """만기(엔트리+hours) 이후 첫 번째 bar의 '종가' 기준으로 청산될 인덱스(그 bar의 close 사용)."""
    target = entry_ts + pd.Timedelta(hours=hours)
    # 만기 시각을 포함하는 bar의 close를 쓰되, 구현상 target 이후 첫 bar의 인덱스 - 1 을 종가로 본다.
    ts = ohlcv["ts"].to_numpy()
    j = int(np.searchsorted(ts, np.datetime64(target), side="right"))
    # j가 0이면(데이터 너무 짧음) 첫 바 종가라도
    j = max(0, min(j, len(ohlcv)-1))
    return j

def simulate(symbol: str, sig_ts_list: List[pd.Timestamp], tp: float, sl: float) -> Tuple[List[Dict], Dict]:
    ohlcv = load_ohlcv(symbol, TIMEFRAME)
    trades = []
    open_until = None  # (entry_end_ts) 심볼당 포지션 오픈 막기용
    for sig_ts in sig_ts_list:
        # 이미 포지션 보유 중(만료 전)이라면 skip
        if open_until is not None and sig_ts < open_until:
            continue

        i = next_bar_open_idx(ohlcv, sig_ts)
        if i >= len(ohlcv):
            continue
        entry_time = ohlcv["ts"].iloc[i]
        entry_price = float(ohlcv["open"].iloc[i])
        # 수수료 진입 반영
        entry_price_fee = entry_price * (1 + FEE_ENTRY)

        j_end = expiry_close_idx(ohlcv, entry_time, EXPIRY_H)
        # 만기 구간 스캔(i..j_end): intrabar TP/SL 히트 체크
        exit_idx = None
        exit_reason = None
        for k in range(i, j_end+1):
            hi = float(ohlcv["high"].iloc[k])
            lo = float(ohlcv["low"].iloc[k])
            # Long 기준
            if hi >= entry_price_fee * (1 + tp):
                exit_idx = k
                exit_reason = "TP"
                break
            if lo <= entry_price_fee * (1 + sl):
                exit_idx = k
                exit_reason = "SL"
                break
        if exit_idx is None:
            exit_idx = j_end
            exit_reason = "EXP"

        exit_time = ohlcv["ts"].iloc[exit_idx]
        # 청산가격(종가 기준) + 수수료
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
        open_until = exit_time  # 이 시간 전까지는 재진입 금지


    # 집계
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

    # TV는 event 다양(예: price_in_box, level2_detected 등) → 모두 Long 엔트리 후보로 간주
    # 필요시 이벤트 필터 좁히세요.
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
        print(f"[BT-TV] {name} done. processed_signals={processed}")
    pd.DataFrame(rows_results).to_csv(OUT_RESULTS, index=False)
    pd.DataFrame(rows_trades).to_csv(OUT_TRADES, index=False)
    print(f"[BT-TV] ✅ 완료: results -> {OUT_RESULTS} , trades -> {OUT_TRADES} (took {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
