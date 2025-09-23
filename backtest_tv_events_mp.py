#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TV 이벤트(backtest) 멀티프로세싱 버전 (안정화)
- 이벤트 그룹: detected / price_in_box / box_breakout / line_breakout
- 만료: 다중 (예: 0.5h,1h,2h,4h,8h)
- TP / SL, 수수료(왕복) 적용
- signals_tv.csv 또는 signals_tv_enriched.csv 모두 지원
  * enriched에 distance_pct 있으면 --dist-max 로 필터
- UTC/타임존 안전화, 빈 OHLCV/결측 방어
- Windows multiprocessing(spawn) 안정화
"""

from __future__ import annotations
import argparse
import math
import os
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# sr_robust.py: 사용자 제공 유틸 (동일 폴더/루트에 배치)
# 기대 인터페이스: get_ohlcv(symbol: str, timeframe: str) -> DataFrame[ts,open,high,low,close], ts는 tz-aware(UTC)
try:
    from sr_robust import get_ohlcv  # type: ignore
except Exception:
    get_ohlcv = None  # 없을 때를 대비해 로컬 CSV 로더만 사용

# -----------------------------
# 공용 유틸 (UTC/타임존 & dtype)
# -----------------------------

def to_utc_series(obj: Iterable) -> pd.Series:
    """임의의 ts 입력을 tz-aware UTC pandas Series로 강제 변환."""
    s = pd.to_datetime(obj, utc=True, errors="coerce")
    # 이미 tz-aware(UTC). 비교/검색은 numpy datetime64[ns]가 편하므로 그대로 유지.
    return s

def utc_ns_array(obj: Iterable) -> np.ndarray:
    """UTC 시리즈를 int64 나노초 epoch 배열로 변환."""
    s = to_utc_series(obj)
    # pandas 경고 없이 안전 변환
    return s.view("int64").to_numpy()

def parse_hours_list(expiries: str) -> List[float]:
    out: List[float] = []
    for tok in expiries.split(","):
        tok = tok.strip().lower()
        if tok.endswith("h"):
            out.append(float(tok[:-1]))
        elif tok.endswith("m"):
            out.append(float(tok[:-1]) / 60.0)
        else:
            out.append(float(tok))  # 시간(float)로 해석
    return out

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

# -----------------------------
# OHLCV 로드 (sr_robust 우선)
# -----------------------------

def load_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    우선 sr_robust.get_ohlcv 사용.
    없으면 ./data/ohlcv/{symbol}_{timeframe}.csv 를 시도.
    ts는 tz-aware(UTC)로 보정.
    """
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
    missing = need - set(map(str, df.columns))
    if missing:
        return None

    df = df.copy()
    # tz-aware(UTC) 강제
    df["ts"] = to_utc_series(df["ts"])
    # 결측/비정상 제거
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)

    return df[["ts", "open", "high", "low", "close"]]

# -----------------------------
# 시뮬레이션 로직
# -----------------------------

def search_idx_right(ts_ns: np.ndarray, key_ns: int) -> int:
    """np.searchsorted(side='right')"""
    return int(np.searchsorted(ts_ns, key_ns, side="right"))

def find_bar_index(ts_ns: np.ndarray, key_ns: int) -> int:
    """key 시점이 포함된 '이전 완성봉' 인덱스(없으면 -1)."""
    i = search_idx_right(ts_ns, key_ns) - 1
    return i

def simulate_symbol(symbol: str,
                    ohlcv: pd.DataFrame,
                    sig_rows: pd.DataFrame,
                    timeframe: str,
                    tp_pct: float,
                    sl_pct: float,
                    fee_rt: float,
                    expiries_h: List[float]) -> pd.DataFrame:
    """
    한 심볼에 대해 이벤트별 백테스트를 수행하고 트레이드 테이블 반환.
    """
    if ohlcv is None or ohlcv.empty or sig_rows is None or sig_rows.empty:
        return pd.DataFrame()

    ts_ns = utc_ns_array(ohlcv["ts"])
    closes = ohlcv["close"].to_numpy(dtype=float)
    highs  = ohlcv["high"].to_numpy(dtype=float)
    lows   = ohlcv["low"].to_numpy(dtype=float)

    out = []

    for s in sig_rows.itertuples(index=False):
        # 필수 컬럼 파싱
        # enriched / original 모두 대응
        sig_ts = getattr(s, "ts")
        event  = getattr(s, "event", None)
        side   = getattr(s, "side", None)
        symbol_row = getattr(s, "symbol", symbol)

        if pd.isna(sig_ts):
            continue
        sig_ns = to_utc_series([sig_ts]).view("int64")[0]

        i_sig = find_bar_index(ts_ns, sig_ns)
        if i_sig < 0 or i_sig >= len(ohlcv) - 1:
            continue

        entry_i = i_sig + 1  # 다음 봉 시가/혹은 close-entry 대신 여기선 close 기준 진입가로 단일화
        entry_px = float(ohlcv.loc[entry_i, "close"])

        # 방향: resistance=롱 금지? 이번 모델은 롱 온리라 가정 -> support 계열만?
        # 여기서는 폴 정의와 무관하게 '롱 온리'로 통일
        # (필요시 event/side 조건을 더 좁혀도 됨)
        # TP/SL 기준가
        tp_price = entry_px * (1.0 + tp_pct / 100.0)
        sl_price = entry_px * (1.0 - sl_pct / 100.0)

        fee_entry = entry_px * fee_rt
        # 만료 루프
        for eh in expiries_h:
            expiry_ns = sig_ns + int(eh * 3600.0 * 1e9)

            # 시뮬레이션 구간: entry_i+1 ~ 만료 전까지
            # 봉 인덱스 탐색
            j_start = entry_i + 1
            j_end = search_idx_right(ts_ns, expiry_ns)  # 만료 봉 시작 전까지
            if j_start >= j_end:
                # 다음 봉 없이 바로 만료
                exit_px = float(ohlcv.loc[min(entry_i, len(ohlcv)-1), "close"])
                pnl = (exit_px - entry_px) / entry_px - fee_rt  # 왕복
                out.append({
                    "symbol": symbol_row,
                    "event": event,
                    "expiry_h": eh,
                    "entry_ts": pd.to_datetime(ts_ns[entry_i]),
                    "exit_ts": pd.to_datetime(ts_ns[min(entry_i, len(ohlcv)-1)]),
                    "entry": entry_px,
                    "exit": exit_px,
                    "tp": tp_pct,
                    "sl": sl_pct,
                    "net": pnl,
                })
                continue

            hit = None
            for k in range(j_start, j_end):
                # 고가로 TP, 저가로 SL 체크 (롱)
                if highs[k] >= tp_price:
                    hit = ("TP", k)
                    break
                if lows[k] <= sl_price:
                    hit = ("SL", k)
                    break

            if hit is None:
                # 만료 청산: 마지막 가용 봉 종가
                k = j_end - 1
                exit_px = float(closes[k])
                pnl = (exit_px - entry_px) / entry_px - fee_rt
                out.append({
                    "symbol": symbol_row,
                    "event": event,
                    "expiry_h": eh,
                    "entry_ts": pd.to_datetime(ts_ns[entry_i]),
                    "exit_ts": pd.to_datetime(ts_ns[k]),
                    "entry": entry_px,
                    "exit": exit_px,
                    "tp": tp_pct,
                    "sl": sl_pct,
                    "net": pnl,
                })
            else:
                kind, k = hit
                exit_px = tp_price if kind == "TP" else sl_price
                pnl = (exit_px - entry_px) / entry_px - fee_rt
                out.append({
                    "symbol": symbol_row,
                    "event": event,
                    "expiry_h": eh,
                    "entry_ts": pd.to_datetime(ts_ns[entry_i]),
                    "exit_ts": pd.to_datetime(ts_ns[k]),
                    "entry": entry_px,
                    "exit": exit_px,
                    "tp": tp_pct,
                    "sl": sl_pct,
                    "net": pnl,
                })

    return pd.DataFrame(out)

# -----------------------------
# 통계
# -----------------------------

def agg_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    win = (df["net"] > 0).sum()
    total = len(df)
    wr = win / total if total else 0.0
    return pd.DataFrame([{
        "trades": float(total),
        "win_rate": wr,
        "avg_net": df["net"].mean(),
        "median_net": df["net"].median(),
        "total_net": df["net"].sum(),
    }])

# -----------------------------
# 메인
# -----------------------------

def run_group(df_sig: pd.DataFrame,
              group_name: str,
              timeframe: str,
              tp: float,
              sl: float,
              fee: float,
              expiries_h: List[float],
              procs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    이벤트 그룹명으로 필터 후, 심볼별 멀티프로세싱 수행.
    """
    sub = df_sig[df_sig["event"] == group_name].copy()
    if sub.empty:
        print(f"[BT][{group_name}] no tasks.")
        return pd.DataFrame(), pd.DataFrame()

    symbols = sorted(sub["symbol"].dropna().unique().tolist())
    tasks = []
    for sym in symbols:
        rows = sub[sub["symbol"] == sym].copy()
        tasks.append((sym, rows))

    print(f"[BT][{group_name}] start: symbols={len(symbols)} rows={len(sub)} tasks={len(tasks)} procs={procs}")

    # 미리 OHLCV 로드해서 각 워커에 전달(피클 안정성 위해 심플 타입만 전달)
    ohlcvs: Dict[str, Optional[pd.DataFrame]] = {}
    for sym in symbols:
        df = load_ohlcv(sym, timeframe)
        if df is None or df.empty:
            print(f"[{sym}] get_ohlcv returned empty.")
        ohlcvs[sym] = df

    # 멀티프로세싱
    from multiprocessing import get_context
    ctx = get_context("spawn")
    parts: List[pd.DataFrame] = []
    with ctx.Pool(processes=procs, maxtasksperchild=100) as pool:
        # starmap에 넘길 튜플 준비
        job_args = []
        for sym, rows in tasks:
            job_args.append((
                sym,
                ohlcvs[sym],
                rows,
                timeframe,
                tp,
                sl,
                fee,
                expiries_h,
            ))
        if job_args:
            parts = pool.starmap(simulate_symbol, job_args)
        else:
            parts = []

    trades = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    trades["group"] = group_name

    stats = trades.groupby(["group", "expiry_h"], as_index=False, dropna=False).apply(agg_stats)
    return trades, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv 또는 signals_tv_enriched.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--dist-max", type=float, default=None, help="enriched의 distance_pct 상한. 없으면 무시")
    ap.add_argument("--outdir", default="./logs")
    args = ap.parse_args()

    df = pd.read_csv(args.signals)
    # 필수 컬럼 표준화/보정
    need_cols = ["ts", "event", "symbol"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"signals 파일에 '{c}' 컬럼이 없습니다.")
    df["ts"] = to_utc_series(df["ts"])
    # distance_pct 필터(있을 때만)
    if args.dist_max is not None and "distance_pct" in df.columns:
        n0 = len(df)
        df = df[df["distance_pct"].apply(safe_float).astype(float) <= float(args.dist_max)].copy()
        print(f"[BT] distance_pct filter {args.dist_max}: {n0}->{len(df)} rows")

    df = df.dropna(subset=["ts", "event", "symbol"]).reset_index(drop=True)
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    print(f"[BT] signals rows={len(df)}, symbols={len(symbols)}, timeframe={args.timeframe}")

    expiries_h = parse_hours_list(args.expiries)

    os.makedirs(args.outdir, exist_ok=True)

    summary_parts = []
    trades_all = []

    for grp in ["detected", "price_in_box", "box_breakout", "line_breakout"]:
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl, args.fee, expiries_h, args.procs)
        # 저장
        if not tr.empty:
            tr_path = os.path.join(args.outdir, f"bt_tv_events_trades_{grp}.csv")
            tr.to_csv(tr_path, index=False)
            print(f"[BT][{grp}] trades -> {tr_path} (rows={len(tr)})")
        if not st.empty:
            st_path = os.path.join(args.outdir, f"bt_tv_events_stats_{grp}.csv")
            st.to_csv(st_path, index=False)
            print(f"[BT][{grp}] stats  -> {st_path} (rows={len(st)})")
        if not st.empty:
            summary_parts.append(st)
        if not tr.empty:
            trades_all.append(tr)

    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame(
        columns=["group", "expiry_h", "trades", "win_rate", "avg_net", "median_net", "total_net"]
    )

    if not summary.empty:
        print("\n=== Summary (by event group & expiry) ===")
        cols = ["group", "expiry_h", "trades", "win_rate", "avg_net", "median_net", "total_net"]
        print(summary[cols].to_string(index=False))

    # 전체 합본 저장
    if trades_all:
        all_trades = pd.concat(trades_all, ignore_index=True)
        all_path = os.path.join(args.outdir, "bt_tv_events_trades.csv")
        all_trades.to_csv(all_path, index=False)
        print(f"\n[BT] saved -> {all_path} (rows={len(all_trades)})")

    sum_path = os.path.join(args.outdir, "bt_tv_events_stats.csv")
    summary.to_csv(sum_path, index=False)
    print(f"[BT] saved -> {sum_path} (rows={len(summary)})")

if __name__ == "__main__":
    # Windows 안정화
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()