# -*- coding: utf-8 -*-
"""
backtest_tv_events_mp.py
- TradingView(폴 지표) 이벤트(Detected / Price in Box / Box Breakout / Line Breakout) 기반
  롱 온리 백테스트. 멀티프로세싱 지원.
- signals_tv_enriched.csv의 distance_pct 필터를 적용 가능.

주요 패치(안정화):
  (1) 빈 결과 안전 처리: simulate_symbol/concat에서 빈 DF면 에러 없이 통과
  (2) OHLCV 경로/패턴/TZ를 CLI로 주입 → get_ohlcv empty 감소
  (3) 멀티프로세싱에서 안전한 concat
  (4) 타임스탬프/타임프레임 정규화
  (5) 요약/저장 로깅 강화

출력:
  outdir/
    bt_tv_events_trades_<group>.csv
    bt_tv_events_stats_<group>.csv
    bt_tv_events_trades.csv         (전체)
    bt_tv_events_stats.csv          (전체 요약)

주의:
  - Upbit는 숏 미지원 → 롱 온리 백테스트
  - TP/SL/fee는 %단위로 입력 (예: --tp 1.5 → +1.5%)
"""

from __future__ import annotations
import os
import re
import math
import time
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# 프로젝트 의존
from sr_engine.data import get_ohlcv

# -----------------------------
# 공통 상수/유틸 (패치 A)
# -----------------------------
TRADE_COLUMNS = [
    "group","symbol","timeframe","event_ts","entry_ts","exit_ts","expiry_h",
    "side","entry","tp","sl","fee","pnl","pnl_pct","hit","reason"
]

def empty_trades_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS)

def ts_to_ns(s: pd.Series) -> np.ndarray:
    """tz-aware/naive 모두 허용, UTC 기준 ns 정수 배열로 변환."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    # pandas 2.2: view 경고 회피 → astype 가능. (값 보존용)
    return t.view("int64").to_numpy()

def parse_expiries(s: str) -> List[float]:
    """'0.5h,1h,2h' → [0.5,1.0,2.0] (시간 단위)"""
    out = []
    for part in str(s).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part.endswith("h"):
            out.append(float(part[:-1]))
        elif part.endswith("m"):
            out.append(float(part[:-1]) / 60.0)
        else:
            # 숫자만 오면 시간으로 간주
            out.append(float(part))
    return out

def ensure_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV에 tz-aware 'ts' 보장."""
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("OHLCV must have DatetimeIndex or 'ts' column")
        ts = df.index
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    out = df.copy()
    out["ts"] = ts
    return out.reset_index(drop=True)

def load_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """get_ohlcv 래퍼. 비정상/빈 결과는 None."""
    try:
        df = get_ohlcv(symbol, timeframe)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    try:
        df = ensure_ts_col(df)
        keep = ["ts","open","high","low","close","volume"]
        for k in keep:
            if k not in df.columns:
                # 누락 컬럼은 만들어 두되, 필수(ts/close 등) 없으면 아래 dropna에서 제거됨
                df[k] = np.nan
        df = df[keep].dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def idx_of_bar(ts: np.ndarray, key_ns: int) -> int:
    """봉 인덱스: key 시각이 속한(혹은 직전) 봉의 index (없으면 -1)."""
    j = int(np.searchsorted(ts, np.int64(key_ns), side="right")) - 1
    return j

# -----------------------------
# 시뮬레이션(롱 온리)
# -----------------------------
def simulate_symbol(symbol: str,
                    timeframe: str,
                    df_sig: pd.DataFrame,
                    tp_pct: float,
                    sl_pct: float,
                    fee_rt: float,
                    expiries_h: List[float]) -> pd.DataFrame:
    """
    (패치 B) OHLCV/신호가 없으면 항상 빈 DF 반환 → 상위 concat 안전.
    엔트리: 신호 발생 봉의 종가로 진입(롱), 유효기간 만료/TP/SL로 청산.
    """
    if df_sig is None or df_sig.empty:
        return empty_trades_df()

    ohlcv = load_ohlcv(symbol, timeframe)
    if ohlcv is None or ohlcv.empty:
        # 호출측에서 로깅하므로 여기선 빈 DF만 반환
        return empty_trades_df()

    ts64 = ts_to_ns(ohlcv["ts"])
    hi = ohlcv["high"].to_numpy(dtype=float)
    lo = ohlcv["low"].to_numpy(dtype=float)
    cl = ohlcv["close"].to_numpy(dtype=float)

    trades = []

    for row in df_sig.itertuples(index=False):
        sig_ts = pd.to_datetime(getattr(row, "ts"), utc=True, errors="coerce")
        if pd.isna(sig_ts):
            continue
        i_sig = idx_of_bar(ts64, int(sig_ts.value))
        if i_sig < 0:
            continue

        entry_idx = i_sig  # 같은 봉 종가 진입
        entry_ts  = ohlcv["ts"].iloc[entry_idx]
        entry     = float(cl[entry_idx])

        # 롱 기준 TP/SL 가격
        tp_px = entry * (1.0 + tp_pct)
        sl_px = entry * (1.0 - sl_pct)

        for eh in expiries_h:
            # 만료 시간 경계
            expiry_limit_ts = entry_ts + pd.Timedelta(hours=float(eh))
            # 탐색 구간 (다음 봉부터 만료 직전/직후 포함)
            # 만료 시점보다 작거나 같은 마지막 봉을 찾는다
            j0 = entry_idx + 1
            if j0 >= len(ohlcv):
                # 다음 봉이 없으면 만료 청산(=진입과 동일 시가정)
                exit_ts = entry_ts
                exit_px = entry
                pnl = (exit_px - entry) / entry - fee_rt
                trades.append({
                    "group": getattr(row, "group", ""),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "event_ts": sig_ts,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "expiry_h": float(eh),
                    "side": "long",
                    "entry": entry,
                    "tp": tp_pct * 100.0,
                    "sl": sl_pct * 100.0,
                    "fee": fee_rt,
                    "pnl": pnl,
                    "pnl_pct": pnl,
                    "hit": "expiry",
                    "reason": "no_next_bar",
                })
                continue

            # 만료 경계 인덱스
            expiry_idx = int(np.searchsorted(ts64, np.int64(expiry_limit_ts.value), side="right")) - 1
            expiry_idx = max(expiry_idx, j0 - 1)
            expiry_idx = min(expiry_idx, len(ohlcv) - 1)
            if expiry_idx < j0:
                # 만료가 다음봉보다 빠른 경우(아주 짧은 만료 등)
                exit_ts = entry_ts
                exit_px = entry
                pnl = (exit_px - entry) / entry - fee_rt
                trades.append({
                    "group": getattr(row, "group", ""),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "event_ts": sig_ts,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "expiry_h": float(eh),
                    "side": "long",
                    "entry": entry,
                    "tp": tp_pct * 100.0,
                    "sl": sl_pct * 100.0,
                    "fee": fee_rt,
                    "pnl": pnl,
                    "pnl_pct": pnl,
                    "hit": "expiry",
                    "reason": "expiry_before_next",
                })
                continue

            # 구간 스캔
            hit = None
            exit_idx = expiry_idx
            for k in range(j0, expiry_idx + 1):
                # 봉 내 TP/SL 충족 여부 (롱 기준: 고가로 TP, 저가로 SL)
                if hi[k] >= tp_px:
                    hit = "tp"
                    exit_idx = k
                    break
                if lo[k] <= sl_px:
                    hit = "sl"
                    exit_idx = k
                    break

            exit_ts = ohlcv["ts"].iloc[exit_idx]
            exit_px = float(cl[exit_idx])

            if hit == "tp":
                pnl = (tp_px - entry) / entry - fee_rt
            elif hit == "sl":
                pnl = (sl_px - entry) / entry - fee_rt
            else:
                pnl = (exit_px - entry) / entry - fee_rt
                hit = "expiry"

            trades.append({
                "group": getattr(row, "group", ""),
                "symbol": symbol,
                "timeframe": timeframe,
                "event_ts": sig_ts,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "expiry_h": float(eh),
                "side": "long",
                "entry": entry,
                "tp": tp_pct * 100.0,
                "sl": sl_pct * 100.0,
                "fee": fee_rt,
                "pnl": pnl,
                "pnl_pct": pnl,
                "hit": hit,
                "reason": "",
            })

    if not trades:
        return empty_trades_df()
    return pd.DataFrame(trades, columns=TRADE_COLUMNS)

# -----------------------------
# 집계/요약
# -----------------------------
def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["group","expiry_h","trades","win_rate","avg_net","median_net","total_net"])
    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        win = (g["pnl_pct"] > 0).sum()
        return pd.Series({
            "trades": float(n),
            "win_rate": float(win) / n if n else 0.0,
            "avg_net": float(g["pnl_pct"].mean()),
            "median_net": float(g["pnl_pct"].median()),
            "total_net": float(g["pnl_pct"].sum()),
        })
    return trades.groupby(["group","expiry_h"], as_index=False, dropna=False).apply(agg)

# -----------------------------
# 그룹 실행 (패치 C)
# -----------------------------
def run_group(df_all: pd.DataFrame,
              group_name: str,
              timeframe: str,
              tp: float,
              sl: float,
              fee_rt: float,
              expiries_h: List[float],
              procs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_all[df_all["group"] == group_name].copy()
    if df.empty:
        print(f"[BT][{group_name}] no tasks.")
        return empty_trades_df(), pd.DataFrame()

    # 심볼별 작업 준비
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    tasks = []
    for sym in symbols:
        rows = df[df["symbol"] == sym].copy()
        if rows.empty:
            continue
        tasks.append((sym, timeframe, rows, tp/100.0, sl/100.0, fee_rt, expiries_h))

    print(f"[BT][{group_name}] start: symbols={len(symbols)} rows={len(df)} tasks={len(tasks)} procs={procs}")
    trades = empty_trades_df()

    if not tasks:
        print(f"[BT][{group_name}] no tasks.")
        return trades, pd.DataFrame()

    if procs and procs > 1:
        with Pool(processes=min(procs, cpu_count())) as pool:
            parts = pool.starmap(simulate_symbol, tasks)
    else:
        parts = [simulate_symbol(*t) for t in tasks]

    frames = [p for p in parts if p is not None and not p.empty]
    if not frames:
        print(f"[BT][{group_name}] no trades (all empty).")
        return empty_trades_df(), pd.DataFrame()

    trades = pd.concat(frames, ignore_index=True)
    stats = summarize(trades)
    return trades, stats

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv_enriched.csv (distance_pct 포함) / 또는 signals_tv.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--procs", type=int, default=max(1, cpu_count()//2))
    ap.add_argument("--dist-max", type=float, default=None, help="distance_pct 상한(%). 예: 0.02 → 0.02%")
    ap.add_argument("--outdir", default="./logs")

    # (패치 D) OHLCV 경로/패턴/TZ 주입 옵션
    ap.add_argument("--ohlcv-roots", default=os.getenv("OHLCV_ROOTS", ""))
    ap.add_argument("--ohlcv-patterns", default=os.getenv("OHLCV_PATTERNS", ""))
    ap.add_argument("--assume-ohlcv-tz", default=os.getenv("ASSUME_OHLCV_TZ", "UTC"))

    args = ap.parse_args()

    # (패치 E) get_ohlcv가 참조할 환경변수 주입
    if args.ohlcv_roots:
        os.environ["OHLCV_ROOTS"] = args.ohlcv_roots
    if args.ohlcv_patterns:
        os.environ["OHLCV_PATTERNS"] = args.ohlcv_patterns
    if args.assume_ohlcv_tz:
        os.environ["ASSUME_OHLCV_TZ"] = args.assume_ohlcv_tz

    expiries_h = parse_expiries(args.expiries)

    if not os.path.exists(args.signals):
        raise FileNotFoundError(args.signals)

    df = pd