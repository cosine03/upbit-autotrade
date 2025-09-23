# -*- coding: utf-8 -*-
r"""
TV signals MP backtest (entry rule = price <= signal_price(1m), long-only)
Usage (예):
  python backtest_tv_entry_close_mp.py .\logs\signals_tv.csv --group alt --expiry 4h,8h --procs 8 --rate 4

옵션:
  --group {all,major,alt} : 심볼 그룹 필터 (기본 all)
  --expiry {4h,8h,...}    : 만기 시간들 콤마(,)로 복수 지정
  --procs N               : 프로세스 수
  --rate K                : 동시 API 호출 상한(세마포어) → pyupbit 폭주 방지
"""

import os, sys, time, math, random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# 외부 API
import pyupbit
from multiprocessing import Pool, Semaphore, get_context

# ---------- 설정 ----------
MAJORS = {"KRW-BTC", "KRW-ETH"}
DEFAULT_TIMEFRAME = "15m"
DEFAULT_COUNT = 400  # 여유롭게
FEE_RT = 0.001  # 왕복 0.1%

# ---------- 유틸: TS 정규화 ----------
def to_utc_ts(x) -> pd.Timestamp:
    """입력(문자열/np.datetime64/Timestamp)을 UTC tz-aware Timestamp로."""
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    # pandas가 utc=True로 만들면 이미 tz-aware이다.
    return ts

def ts_series_to_ns(s: pd.Series) -> np.ndarray:
    """tz-aware Datetime Series -> int64 ns numpy array"""
    # pandas의 .view('int64')는 최신판에서 경고가 있을 수 있어 astype 사용
    return s.view("int64").to_numpy()

def dt_to_ns(x: pd.Timestamp) -> int:
    return int(to_utc_ts(x).value)

# ---------- OHLCV 안전 로더 ----------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    if "index" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"index": "ts"})
    if "time" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"time": "ts"})
    if "ts" not in df.columns:
        # pyupbit는 기본이 DatetimeIndex, 위에서 대부분 잡힌다
        raise ValueError("No 'ts' column after reset_index()")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    need_cols = {"open","high","low","close"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"OHLCV columns missing: need {need_cols}, got {df.columns}")
    return df[["ts","open","high","low","close"]].sort_values("ts").reset_index(drop=True)

def get_ohlcv_safe(symbol: str, tf: str, count: int, sem: Optional[Semaphore], retries: int = 3, base_sleep: float = 0.35) -> Optional[pd.DataFrame]:
    for i in range(retries):
        try:
            if sem is not None:
                with sem:
                    df = pyupbit.get_ohlcv(ticker=symbol, interval=tf, count=count)
            else:
                df = pyupbit.get_ohlcv(ticker=symbol, interval=tf, count=count)
            if df is None or len(df) == 0:
                raise ValueError("empty ohlcv")
            return normalize_ohlcv(df)
        except Exception as e:
            if i == retries - 1:
                print(f"[{symbol}] OHLCV load error: {e!r}")
                return None
            time.sleep(base_sleep + random.random() * 0.4)
    return None

# ---------- 엔트리/익절/손절 ----------

def next_bar_open_idx(ohlcv: pd.DataFrame, signal_ts: pd.Timestamp) -> int:
    """
    시그널 직후 첫 오픈 인덱스.
    ohlcv["ts"] 는 tz-aware, search는 ns(int64)로.
    """
    ts_ns = ts_series_to_ns(ohlcv["ts"])
    sig_ns = to_utc_ts(signal_ts).value
    # right: signal 시점 이후 첫 캔들
    return int(np.searchsorted(ts_ns, sig_ns, side="right"))

def price_at_1m(symbol: str, signal_ts: pd.Timestamp, sem: Optional[Semaphore]) -> Optional[float]:
    """
    시그널 발생 '당시' 가격(1분봉 close)을 대략 캡처.
    실패하면 None.
    """
    df1 = get_ohlcv_safe(symbol, "minute1", 10, sem)  # 10개면 ±10분 커버
    if df1 is None:
        return None
    # 신호 시각과 가장 가까운 1분봉 close를 사용(이전 1분봉 우선)
    idx = next_bar_open_idx(df1, signal_ts) - 1
    if idx < 0:
        idx = 0
    if idx >= len(df1):
        idx = len(df1) - 1
    return float(df1["close"].iloc[idx])

@dataclass
class Task:
    symbol: str
    rows: List[dict]
    timeframe: str
    tp: float
    sl: float
    expiry_h: int
    sem: Optional[Semaphore]

def simulate_symbol(task: Task) -> pd.DataFrame:
    sym, rows, timeframe, tp, sl, expiry_h, sem = (
        task.symbol, task.rows, task.timeframe, task.tp, task.sl, task.expiry_h, task.sem
    )

    # 15m OHLCV
    ohlcv = get_ohlcv_safe(sym, timeframe, DEFAULT_COUNT, sem)
    if ohlcv is None or len(ohlcv) < 10:
        return pd.DataFrame(columns=["symbol","ts","entry_ts","exit_ts","entry","exit","ret","tp","sl","expiry_h","reason"])

    ts_ns = ts_series_to_ns(ohlcv["ts"])
    out_rows = []

    for s in rows:
        sig_ts = to_utc_ts(s["ts"])
        if sig_ts is None:
            continue

        # 1) 엔트리 조건: 이후에 price <= signal_price(1m) 발생 시 진입
        sig_px = price_at_1m(sym, sig_ts, sem)
        # 진입 후보 바 → 시그널 직후 첫 캔들부터 탐색
        i0 = next_bar_open_idx(ohlcv, sig_ts)
        if i0 >= len(ohlcv):
            continue

        enter_idx = None
        for i in range(i0, len(ohlcv)):
            # low <= sig_px (동등 포함) 이면 그 캔들 오픈가로 진입
            if sig_px is not None and float(ohlcv["low"].iloc[i]) <= sig_px:
                enter_idx = i
                break
        if enter_idx is None:
            continue

        entry_ts = ohlcv["ts"].iloc[enter_idx]
        entry_px = float(ohlcv["open"].iloc[enter_idx])

        # 2) 익절/손절/만기
        #   만기 인덱스: 만기 시각의 "바로 이전"까지 검사하고 만기 시각에 종가 청산
        expiry_ts = entry_ts + pd.Timedelta(hours=expiry_h)
        exp_idx = int(np.searchsorted(ts_ns, expiry_ts.value, side="left"))
        if exp_idx <= enter_idx:
            exp_idx = enter_idx + 1
        if exp_idx > len(ohlcv) - 1:
            exp_idx = len(ohlcv) - 1

        take = entry_px * (1.0 + tp/100.0)
        stop = entry_px * (1.0 - sl/100.0)

        exit_reason = "expiry"
        exit_idx = exp_idx
        # 엔트리 이후부터 만기 직전까지 고/저 체크
        for i in range(enter_idx + 1, exp_idx + 1):
            hi = float(ohlcv["high"].iloc[i])
            lo = float(ohlcv["low"].iloc[i])
            if hi >= take:
                exit_reason = "tp"
                exit_idx = i
                break
            if lo <= stop:
                exit_reason = "sl"
                exit_idx = i
                break

        exit_px = float(ohlcv["close"].iloc[exit_idx])
        gross = (exit_px / entry_px) - 1.0
        net = gross - FEE_RT

        out_rows.append({
            "symbol": sym,
            "ts": sig_ts.isoformat(),
            "entry_ts": entry_ts.isoformat(),
            "exit_ts": ohlcv["ts"].iloc[exit_idx].isoformat(),
            "entry": entry_px,
            "exit": exit_px,
            "ret": net,
            "tp": tp,
            "sl": sl,
            "expiry_h": expiry_h,
            "reason": exit_reason,
        })

    return pd.DataFrame(out_rows)

# ---------- 시그널 로드/그룹 ----------
def load_tv_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 통일 컬럼 기대: ts,event,side,level,touches,symbol,timeframe,extra,source,host,message
    # 혹은 이전 포맷(ts_utc / symbol / ...)
    if "ts" not in df.columns and "ts_utc" in df.columns:
        df = df.rename(columns={"ts_utc": "ts"})
    if "symbol" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","symbol"])
    df["symbol"] = df["symbol"].astype(str)
    return df[["ts","symbol"]].sort_values("ts").reset_index(drop=True)

def split_by_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    group = (group or "all").lower()
    if group == "major":
        return df[df["symbol"].isin(MAJORS)].copy()
    if group == "alt":
        return df[~df["symbol"].isin(MAJORS)].copy()
    return df.copy()

# ---------- 통계 ----------
def agg_stats(g: pd.DataFrame) -> pd.Series:
    trades = float(len(g))
    if trades == 0:
        return pd.Series(dict(trades=0, win_rate=np.nan, avg_net=np.nan, median_net=np.nan, total_net=0.0))
    win = float((g["ret"] > 0).sum())
    return pd.Series(dict(
        trades=trades,
        win_rate=win / trades,
        avg_net=float(g["ret"].mean()),
        median_net=float(g["ret"].median()),
        total_net=float(g["ret"].sum()),
    ))

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals_csv", type=str)
    ap.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    ap.add_argument("--group", default="all", choices=["all","major","alt"])
    ap.add_argument("--expiry", default="4h,8h")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--rate", type=int, default=4, help="동시 API 호출 상한 (권장 3~6)")
    args = ap.parse_args()

    df_sig = split_by_group(load_tv_signals(args.signals_csv), args.group)
    if df_sig.empty:
        print("No signals after grouping.")
        return

    # 만기 리스트 파싱
    expiry_list: List[int] = []
    for x in str(args.expiry).split(","):
        x = x.strip().lower()
        if x.endswith("h"):
            expiry_list.append(int(x[:-1]))
        elif x.isdigit():
            expiry_list.append(int(x))
    expiry_list = sorted(set(expiry_list))

    # 전략 모음
    strats = [
        ("stable", 1.5, 1.0),
        ("aggressive", 2.0, 1.25),
        ("scalp", 1.0, 0.75),
        ("mid_1", 1.25, 1.0),
        ("mid_2", 1.75, 1.25),
        ("tight", 0.8, 0.8),
    ]

    # 심볼별로 시그널 묶기
    tasks: List[Task] = []
    for sym, sub in df_sig.groupby("symbol"):
        rows = [{"ts": t} for t in sub["ts"].tolist()]
        for (name, tp, sl) in strats:
            for exp in expiry_list:
                # 세마포어는 모든 작업에 동일 객체 전달
                tasks.append(Task(symbol=sym, rows=rows, timeframe=args.timeframe,
                                  tp=tp, sl=sl, expiry_h=exp, sem=None))

    # 공용 세마포어 생성(동시 API 호출 제한)
    sem = Semaphore(args.rate)
    for t in tasks:
        t.sem = sem

    ctx = get_context("spawn")  # Windows 안전
    with ctx.Pool(processes=args.procs) as pool:
        parts = pool.map(simulate_symbol, tasks)

    trades = pd.concat([p for p in parts if p is not None and len(p) > 0], ignore_index=True) \
              if parts else pd.DataFrame(columns=["symbol","ts","entry_ts","exit_ts","entry","exit","ret","tp","sl","expiry_h","reason"])

    # 저장 & 요약
    os.makedirs("logs", exist_ok=True)
    trades.to_csv("./logs/bt_tv_entryclose_trades.csv", index=False)

    stats = trades.groupby(["tp","sl","expiry_h"], as_index=False, dropna=False).apply(agg_stats)
    stats = stats.rename(columns={"tp":"tp_pct","sl":"sl_pct"})
    stats["strategy"] = stats.apply(lambda r: f"{r['tp_pct']}/{r['sl_pct']}_{int(r['expiry_h'])}h", axis=1)
    stats = stats[["strategy","trades","win_rate","avg_net","median_net","total_net"]]
    stats.to_csv("./logs/bt_tv_entryclose_stats.csv", index=False)

    print("\n=== TV Backtest (Entry = price <= signal_price(1m) / Long-only / fees=0.1% RT) ===")
    print(f"Trades saved: ./logs/bt_tv_entryclose_trades.csv (rows={len(trades)})")
    print(f"Stats  saved: ./logs/bt_tv_entryclose_stats.csv (rows={len(stats)})\n")
    # 상위 몇 개 출력
    print(stats.sort_values(["expiry_h","avg_net"], ascending=[True,False]).to_string(index=False))

if __name__ == "__main__":
    main()