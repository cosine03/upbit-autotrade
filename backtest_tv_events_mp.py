# -*- coding: utf-8 -*-
"""
backtest_tv_events_mp.py
- TradingView (Paul indicator) events 백테스트 멀티프로세싱 버전
- 외부 sr_engine.simulate 의존 제거: simulate_event_group 로컬 구현
- distance_pct 필터 지원 (--dist-max)
- 롱 전용 (support 이벤트만 체결; resistance는 스킵)

입력: signals_tv_enriched.csv (estimate_tv_levels.py로 생성)
필수 컬럼: ts, event, side, symbol, timeframe, distance_pct(%) 등
"""

import os
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import Tuple, List, Dict, Optional

# OHLCV 로드는 기존 프로젝트의 함수 사용 (있다고 가정)
from sr_engine.data import get_ohlcv


# -------------------- 유틸 --------------------

def ts_to_ns(s: pd.Series) -> np.ndarray:
    """UTC aware datetime 시리즈를 ns int64 배열로 변환 (pandas 2.x 호환)"""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.view("int64").to_numpy()

def timeframe_to_minutes(tf: str) -> int:
    s = tf.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(float(s[:-1]) * 60)
    if s.endswith("d"):
        return int(float(s[:-1]) * 60 * 24)
    return 15  # default

def filter_by_distance(df: pd.DataFrame, dist_max: Optional[float]) -> pd.DataFrame:
    """distance_pct 필터 적용. dist_max는 소수(0.02=2%) 단위"""
    if dist_max is None or "distance_pct" not in df.columns:
        return df
    before = len(df)
    df2 = df[pd.to_numeric(df["distance_pct"], errors="coerce") <= dist_max * 100.0].copy()
    after = len(df2)
    print(f"[BT] distance_pct filter {dist_max}: {before}->{after} rows")
    return df2

def idx_of_bar(ts_arr_ns: np.ndarray, key_ns: np.int64) -> int:
    """시그널 시각이 포함되는 봉의 인덱스(또는 직전 인덱스)를 찾는다."""
    i = int(np.searchsorted(ts_arr_ns, key_ns, side="right")) - 1
    return i

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV df를 표준형으로 보정 + tz-aware ts 컬럼 보장"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            ts = out.index
        else:
            raise ValueError("OHLCV must have DatetimeIndex or 'ts' column")
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        out = out.reset_index(drop=False).rename(columns={out.columns[0]: "ts"})
        out["ts"] = ts
    else:
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    # 필요한 컬럼만
    keep = [c for c in ["ts", "open", "high", "low", "close"] if c in out.columns]
    out = out[keep].dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
    return out


# -------------------- 로컬 시뮬레이션 엔진 --------------------

def simulate_event_group(
    symbol: str,
    ohlcv: pd.DataFrame,
    rows: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float,
    expiries_h: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    간단/안정화 롱전용 엔진:
      - 엔트리: 시그널 봉의 close
      - 방향: side=='support' 만 체결 (resistance는 스킵)
      - TP/SL: 봉별 high/low로 선행충족 체크
      - 만료: expiry_h를 timeframe 분 단위로 변환→N봉 뒤 close 청산
    """
    ohlcv = ensure_ohlcv(ohlcv)
    if ohlcv.empty or rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    ts_ns = ts_to_ns(ohlcv["ts"])
    tf_min = timeframe_to_minutes(str(rows["timeframe"].iloc[0] if "timeframe" in rows.columns else "15m"))

    trades_out: List[Dict] = []

    # 각 시그널 행 처리
    for _, r in rows.iterrows():
        side = str(r.get("side", "") or "").strip().lower()
        if side != "support":
            # 롱 전용: resistance는 스킵
            continue

        sig_ts = pd.to_datetime(r["ts"], utc=True, errors="coerce")
        if pd.isna(sig_ts):
            continue
        i = idx_of_bar(ts_ns, np.int64(sig_ts.value))
        if i < 0 or i >= len(ohlcv):
            continue

        entry = float(ohlcv["close"].iloc[i])
        if not np.isfinite(entry) or entry <= 0:
            continue

        tp_price = entry * (1.0 + tp_pct / 100.0)
        sl_price = entry * (1.0 - sl_pct / 100.0)

        # 각 expiry 별로 결과 생성
        for eh in expiries_h:
            # 만료 봉 수
            bars = max(1, int(round((eh * 60.0) / tf_min)))
            j_end = min(len(ohlcv) - 1, i + bars)

            hit: Optional[str] = None
            exit_px: float = entry

            # i+1 부터 j_end 까지 진행하며 선행 TP/SL 체크
            for j in range(i + 1, j_end + 1):
                hi = float(ohlcv["high"].iloc[j])
                lo = float(ohlcv["low"].iloc[j])

                # TP 선행
                if hi >= tp_price:
                    exit_px = tp_price
                    hit = "tp"
                    break
                # SL 선행
                if lo <= sl_price:
                    exit_px = sl_price
                    hit = "sl"
                    break

            if hit is None:
                # 만료 청산: 마지막 봉 종가
                exit_px = float(ohlcv["close"].iloc[j_end])
                hit = "expiry"

            gross = (exit_px / entry) - 1.0
            net = gross - (fee_rt * 2.0)  # 진입+청산 수수료
            trades_out.append({
                "symbol": symbol,
                "event": r.get("event", ""),
                "side": side,
                "expiry_h": eh,
                "ts_entry": ohlcv["ts"].iloc[i],
                "ts_exit": ohlcv["ts"].iloc[j_end] if hit == "expiry" else ohlcv["ts"].iloc[j if j <= j_end else j_end],
                "entry": entry,
                "exit": exit_px,
                "gross_pct": gross,
                "net_pct": net,
                "result": hit,
            })

    trades_df = pd.DataFrame(trades_out)

    # 그룹별 요약 통계
    if not trades_df.empty:
        def agg(g: pd.DataFrame) -> pd.Series:
            wins = (g["result"] == "tp").sum()
            return pd.Series({
                "trades": len(g),
                "win_rate": wins / max(1, len(g)),
                "avg_net": g["net_pct"].mean(),
                "median_net": g["net_pct"].median(),
                "total_net": g["net_pct"].sum(),
            })

        stats_df = (trades_df
                    .groupby(["event", "expiry_h"], as_index=False)
                    .apply(agg)
                    .reset_index(drop=True))
    else:
        stats_df = pd.DataFrame()

    return trades_df, stats_df


# -------------------- 멀티프로세싱 래퍼 --------------------

def simulate_symbol(symbol: str, rows: pd.DataFrame,
                    timeframe: str, tp: float, sl: float, fee: float,
                    expiries_h: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ohlcv = get_ohlcv(symbol, timeframe)
    if ohlcv is None or len(ohlcv) == 0:
        print(f"[{symbol}] get_ohlcv returned empty.")
        return pd.DataFrame(), pd.DataFrame()

    trades, stats = simulate_event_group(
        symbol=symbol,
        ohlcv=ohlcv,
        rows=rows,
        tp_pct=tp,
        sl_pct=sl,
        fee_rt=fee,
        expiries_h=expiries_h
    )
    return trades, stats


def run_group(df: pd.DataFrame, group: str, timeframe: str,
              tp: float, sl: float, fee: float,
              expiries_h: List[float], procs: int):

    df_g = df[df["event"] == group].copy()
    rows = len(df_g)
    syms = df_g["symbol"].nunique()
    print(f"[BT][{group}] start: symbols={syms} rows={rows} tasks={syms} procs={procs}")

    if rows == 0 or syms == 0:
        print(f"[BT][{group}] no tasks.")
        return pd.DataFrame(), pd.DataFrame()

    tasks = [(sym, df_g[df_g["symbol"] == sym], timeframe, tp, sl, fee, expiries_h)
             for sym in df_g["symbol"].unique()]

    if procs > 1:
        with mp.Pool(processes=procs) as pool:
            parts = pool.starmap(simulate_symbol, tasks)
    else:
        parts = [simulate_symbol(*t) for t in tasks]

    trades = pd.concat([p[0] for p in parts if p[0] is not None and not p[0].empty],
                       ignore_index=True) if parts else pd.DataFrame()
    stats = pd.concat([p[1] for p in parts if p[1] is not None and not p[1].empty],
                      ignore_index=True) if parts else pd.DataFrame()

    if not trades.empty:
        print(f"[BT][{group}] trades -> rows={len(trades)}")
    else:
        print(f"[BT][{group}] no trades.")

    return trades, stats


# -------------------- 메인 --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv_enriched.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h", help="예: 0.5h,1h,2h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dist-max", type=float, default=None, help="distance_pct 필터 (0.02 → 2%)")
    ap.add_argument("--outdir", default="./logs")
    args = ap.parse_args()

    # 입력
    df = pd.read_csv(args.signals)
    print(f"[BT] signals rows={len(df)}, symbols={df['symbol'].nunique()}, timeframe={args.timeframe}")

    # distance_pct 필터
    df = filter_by_distance(df, args.dist_max)

    # expiry 파싱
    expiries_h: List[float] = []
    for e in args.expiries.split(","):
        e = e.strip().lower()
        if not e:
            continue
        if e.endswith("h"):
            expiries_h.append(float(e[:-1]))
        elif e.endswith("m"):
            expiries_h.append(float(e[:-1]) / 60.0)
        elif e.endswith("d"):
            expiries_h.append(float(e[:-1]) * 24.0)
        else:
            # 숫자만 들어오면 시간 단위로 가정
            expiries_h.append(float(e))
    expiries_h = [float(x) for x in expiries_h if x > 0]

    groups = ["detected", "price_in_box", "box_breakout", "line_breakout"]

    all_trades = []
    all_stats = []
    for grp in groups:
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl,
                           args.fee, expiries_h, args.procs)
        if not tr.empty:
            all_trades.append(tr)
        if not st.empty:
            all_stats.append(st)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    # 결과 저장
    os.makedirs(args.outdir, exist_ok=True)
    trades_path = os.path.join(args.outdir, "bt_tv_events_trades.csv")
    stats_path = os.path.join(args.outdir, "bt_tv_events_stats.csv")

    trades.to_csv(trades_path, index=False)
    stats.to_csv(stats_path, index=False)

    print(f"[BT] saved -> {trades_path} (rows={len(trades)})")
    print(f"[BT] saved -> {stats_path} (rows={len(stats)})")


if __name__ == "__main__":
    mp.freeze_support()
    main()