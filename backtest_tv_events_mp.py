# -*- coding: utf-8 -*-
"""
backtest_tv_events_mp.py  (LOCAL_SIM v3)
- TradingView (Paul indicator) 이벤트 백테스트 멀티프로세싱
- 외부 모듈(sr_engine 등) 0% 의존. 로컬 CSV OHLCV만 사용.
- distance_pct 필터(--dist-max) / 만료(0.5h/1h/2h 등) / 수수료 / 멀티프로세싱 지원
- 엔트리=시그널 봉 종가, TP/SL 선행충족, 만료시 종가청산 (롱 전용: support만 진입)

입력: signals_tv_enriched.csv (estimate_tv_levels.py 결과)
필수 컬럼: ts, event, side, symbol, timeframe, distance_pct (단위: %) ...
"""

import os
import re
import argparse
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


# -------------------- 디버그 배너 --------------------
BANNER = "BACKTEST TV EVENTS (LOCAL_SIM v3, no external deps)"
print(f"[BT] {BANNER}")


# -------------------- 공통 유틸 --------------------

def to_utc_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def ts_to_ns(s: pd.Series) -> np.ndarray:
    s = pd.to_datetime(s, utc=True, errors="coerce")
    # pandas 2.x 호환: view 대신 astype도 가능하지만 여기선 view 유지
    return s.view("int64").to_numpy()

def timeframe_to_minutes(tf: str) -> int:
    s = str(tf).strip().lower()
    if s.endswith("m"):
        return int(float(s[:-1]))
    if s.endswith("h"):
        return int(float(s[:-1]) * 60)
    if s.endswith("d"):
        return int(float(s[:-1]) * 60 * 24)
    return 15

def idx_of_bar(ts_arr_ns: np.ndarray, key_ns: np.int64) -> int:
    return int(np.searchsorted(ts_arr_ns, key_ns, side="right")) - 1

def filter_by_distance(df: pd.DataFrame, dist_max: Optional[float]) -> pd.DataFrame:
    if dist_max is None or "distance_pct" not in df.columns:
        return df
    before = len(df)
    # distance_pct는 % 단위(예: 2% -> 2.0) 이므로 dist_max(0.02)를 *100
    df2 = df[pd.to_numeric(df["distance_pct"], errors="coerce") <= dist_max * 100.0].copy()
    after = len(df2)
    print(f"[BT] distance_pct filter {dist_max}: {before}->{after} rows")
    return df2


# -------------------- 로컬 OHLCV 로더 --------------------

CAND_TS_COLS = [
    "ts", "timestamp", "datetime", "date", "time",
    "candle_time", "candle_date_time_kst", "candle_date_time_utc"
]
CAND_OPEN = ["open", "o"]
CAND_HIGH = ["high", "h"]
CAND_LOW  = ["low", "l"]
CAND_CLOSE= ["close", "c"]

def _first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    return None

def _parse_ts_col(df: pd.DataFrame, assume_tz: Optional[str]) -> pd.Series:
    # 1) 후보 컬럼 찾기
    ts_col = None
    for name in CAND_TS_COLS:
        if name in df.columns:
            ts_col = name
            break
        # 대소문자 대응
        for c in df.columns:
            if c.lower() == name:
                ts_col = c
                break
        if ts_col:
            break

    if ts_col is None:
        # 인덱스가 DatetimeIndex면 사용
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
            ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
            return pd.Series(ts, name="ts")
        # epoch(ms) 형식으로 저장된 일반 열이름 추정 실패 시 시도
        raise ValueError("No timestamp column found")

    s = df[ts_col]

    # Upbit kst/utc 특수케이스
    if ts_col.lower() == "candle_date_time_kst":
        ts = pd.to_datetime(s, errors="coerce")
        ts = ts.dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
        return ts
    if ts_col.lower() == "candle_date_time_utc":
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        return ts

    # 숫자형(에폭)인지 문자열(날짜)인지 구분
    if pd.api.types.is_numeric_dtype(s):
        # heuristic: 값 크기로 ms/초 판단
        sr = pd.to_numeric(s, errors="coerce")
        # 크기가 10자리면 초, 13자리면 ms일 가능성
        if sr.dropna().median() > 1e12:  # ms
            ts = pd.to_datetime(sr, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(sr, unit="s",  utc=True, errors="coerce")
        return ts

    # 문자열 날짜
    ts = pd.to_datetime(s, errors="coerce")
    if ts.dt.tz is None:  # tz-naive
        if assume_tz:
            ts = ts.dt.tz_localize(assume_tz).dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts

def _norm_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    oc = _first_col(df, CAND_OPEN)
    hc = _first_col(df, CAND_HIGH)
    lc = _first_col(df, CAND_LOW)
    cc = _first_col(df, CAND_CLOSE)
    if not all([oc, hc, lc, cc]):
        raise ValueError("Missing one of OHLC columns")
    out = pd.DataFrame({
        "open":  pd.to_numeric(df[oc], errors="coerce"),
        "high":  pd.to_numeric(df[hc], errors="coerce"),
        "low":   pd.to_numeric(df[lc], errors="coerce"),
        "close": pd.to_numeric(df[cc], errors="coerce"),
    })
    return out

def _candidate_paths(roots: List[str], patterns: List[str], symbol: str, timeframe: str) -> List[str]:
    out = []
    for r in roots:
        r = r.strip().strip('"')
        if not r:
            continue
        for pat in patterns:
            pat = pat.strip().strip('"')
            if not pat:
                continue
            rel = pat.format(symbol=symbol, timeframe=timeframe)
            p = os.path.join(r, rel) if not os.path.isabs(rel) else rel
            out.append(os.path.normpath(p))
    # 중복 제거(앞선 우선)
    seen = set()
    out2 = []
    for p in out:
        if p not in seen:
            out2.append(p); seen.add(p)
    return out2

def load_ohlcv_local(symbol: str, timeframe: str,
                     roots: List[str], patterns: List[str],
                     assume_tz: Optional[str]) -> pd.DataFrame:
    paths = _candidate_paths(roots, patterns, symbol, timeframe)
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                try:
                    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
                except Exception:
                    continue
            try:
                ts = _parse_ts_col(df, assume_tz)
                ohlc = _norm_ohlcv_cols(df)
                out = pd.DataFrame({"ts": ts})
                out = pd.concat([out, ohlc], axis=1)
                out = out.dropna(subset=["ts", "open", "high", "low", "close"]).copy()
                out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
                out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
                if not out.empty:
                    return out
            except Exception:
                continue
    # 못 찾으면 빈 df
    return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])


# -------------------- 시뮬레이션 엔진 (롱 전용) --------------------

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts","open","high","low","close"])
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            ts = out.index
            ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
            out = out.reset_index(drop=False).rename(columns={out.columns[0]: "ts"})
            out["ts"] = ts
        else:
            raise ValueError("OHLCV must have DatetimeIndex or 'ts' column")
    else:
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    keep = [c for c in ["ts","open","high","low","close"] if c in out.columns]
    out = out[keep].dropna(subset=["ts","open","high","low","close"]).sort_values("ts").reset_index(drop=True)
    return out

def simulate_event_group(symbol: str, ohlcv: pd.DataFrame, rows: pd.DataFrame,
                         tp_pct: float, sl_pct: float, fee_rt: float,
                         expiries_h: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    ohlcv = ensure_ohlcv(ohlcv)
    if ohlcv.empty or rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    ts_ns = ts_to_ns(ohlcv["ts"])
    tf_min = timeframe_to_minutes(str(rows["timeframe"].iloc[0] if "timeframe" in rows.columns else "15m"))

    trades_out: List[Dict] = []

    for _, r in rows.iterrows():
        side = str(r.get("side", "") or "").strip().lower()
        if side != "support":  # 롱 전용
            continue

        sig_ts = to_utc_ts(r["ts"])
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

        for eh in expiries_h:
            bars = max(1, int(round((eh * 60.0) / tf_min)))
            j_end = min(len(ohlcv) - 1, i + bars)

            hit = None
            exit_px = entry

            for j in range(i + 1, j_end + 1):
                hi = float(ohlcv["high"].iloc[j])
                lo = float(ohlcv["low"].iloc[j])
                if hi >= tp_price:
                    exit_px = tp_price; hit = "tp"; break
                if lo <= sl_price:
                    exit_px = sl_price; hit = "sl"; break

            if hit is None:
                exit_px = float(ohlcv["close"].iloc[j_end])
                hit = "expiry"

            gross = (exit_px / entry) - 1.0
            net = gross - (fee_rt * 2.0)

            trades_out.append({
                "symbol": symbol,
                "event": r.get("event", ""),
                "side": side,
                "expiry_h": eh,
                "ts_entry": ohlcv["ts"].iloc[i],
                "ts_exit": ohlcv["ts"].iloc[j_end] if hit == "expiry" else ohlcv["ts"].iloc[j],
                "entry": entry,
                "exit": exit_px,
                "gross_pct": gross,
                "net_pct": net,
                "result": hit,
            })

    trades_df = pd.DataFrame(trades_out)

    if not trades_df.empty:
        def agg(g: pd.DataFrame) -> pd.Series:
            wins = (g["result"] == "tp").sum()
            return pd.Series({
                "trades": int(len(g)),
                "win_rate": float(wins) / max(1, len(g)),
                "avg_net": float(g["net_pct"].mean()),
                "median_net": float(g["net_pct"].median()),
                "total_net": float(g["net_pct"].sum()),
            })
        stats_df = (trades_df.groupby(["event","expiry_h"], as_index=False)
                    .apply(agg).reset_index(drop=True))
    else:
        stats_df = pd.DataFrame(columns=["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"])

    return trades_df, stats_df


# -------------------- 멀티프로세싱 --------------------

def simulate_symbol(symbol: str, rows: pd.DataFrame, timeframe: str,
                    tp: float, sl: float, fee: float, expiries_h: List[float],
                    roots: List[str], patterns: List[str], assume_tz: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    ohlcv = load_ohlcv_local(symbol, timeframe, roots, patterns, assume_tz)
    if ohlcv.empty:
        print(f"[{symbol}] get_ohlcv returned empty.")
        return pd.DataFrame(), pd.DataFrame()

    return simulate_event_group(symbol, ohlcv, rows, tp, sl, fee, expiries_h)


def run_group(df: pd.DataFrame, group: str, timeframe: str, tp: float, sl: float, fee: float,
              expiries_h: List[float], procs: int,
              roots: List[str], patterns: List[str], assume_tz: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_g = df[df["event"] == group].copy()
    rows = len(df_g)
    syms = df_g["symbol"].nunique()
    print(f"[BT][{group}] start: symbols={syms} rows={rows} tasks={syms} procs={procs}")

    if rows == 0 or syms == 0:
        print(f"[BT][{group}] no tasks.")
        return pd.DataFrame(), pd.DataFrame()

    tasks = [(sym,
              df_g[df_g["symbol"] == sym],
              timeframe, tp, sl, fee, expiries_h,
              roots, patterns, assume_tz)
             for sym in df_g["symbol"].unique()]

    if procs > 1:
        with mp.Pool(processes=procs) as pool:
            parts = pool.starmap(simulate_symbol, tasks)
    else:
        parts = [simulate_symbol(*t) for t in tasks]

    trades = pd.concat([p[0] for p in parts if p and not p[0].empty],
                       ignore_index=True) if parts else pd.DataFrame()
    stats  = pd.concat([p[1] for p in parts if p and not p[1].empty],
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

    # 로컬 OHLCV 설정 (estimate_tv_levels.py 와 동일 스타일)
    ap.add_argument("--ohlcv-roots", default=".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv",
                    help="세미콜론(;) 구분 루트 목록")
    ap.add_argument("--ohlcv-patterns",
                    default="data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;ohlcv/{symbol}-{timeframe}.csv;ohlcv/{symbol}_{timeframe}.csv;logs/ohlcv/{symbol}-{timeframe}.csv;logs/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv",
                    help="세미콜론(;) 구분 패턴 목록")
    ap.add_argument("--assume-ohlcv-tz", default=None,
                    help="OHLCV가 tz-naive일 때 가정할 타임존 (예: Asia/Seoul)")

    args = ap.parse_args()

    # 시그널 로드
    df = pd.read_csv(args.signals)
    # 기본 체크
    if "ts" not in df.columns or "symbol" not in df.columns or "event" not in df.columns:
        raise ValueError("signals 파일에 'ts','symbol','event' 컬럼이 필요합니다.")
    if "timeframe" not in df.columns:
        df["timeframe"] = args.timeframe
    print(f"[BT] signals rows={len(df)}, symbols={df['symbol'].nunique()}, timeframe={args.timeframe}")

    # distance 필터
    df = filter_by_distance(df, args.dist_max)

    # 만료 리스트 파싱
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
            expiries_h.append(float(e))
    expiries_h = [x for x in expiries_h if x > 0]

    # 루트/패턴 파싱
    roots = [r.strip().strip('"').replace("\\", "/") for r in args.ohlcv_roots.split(";") if r.strip()]
    patterns = [p.strip().strip('"').replace("\\", "/") for p in args.ohlcv_patterns.split(";") if p.strip()]
    assume_tz = args.assume_ohlcv_tz

    groups = ["detected", "price_in_box", "box_breakout", "line_breakout"]

    all_trades, all_stats = [], []
    for grp in groups:
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl, args.fee,
                           expiries_h, args.procs, roots, patterns, assume_tz)
        if not tr.empty: all_trades.append(tr)
        if not st.empty: all_stats.append(st)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(
        columns=["symbol","event","side","expiry_h","ts_entry","ts_exit","entry","exit","gross_pct","net_pct","result"])
    stats  = pd.concat(all_stats,  ignore_index=True) if all_stats  else pd.DataFrame(
        columns=["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"])

    os.makedirs(args.outdir, exist_ok=True)
    trades_path = os.path.join(args.outdir, "bt_tv_events_trades.csv")
    stats_path  = os.path.join(args.outdir, "bt_tv_events_stats.csv")
    trades.to_csv(trades_path, index=False)
    stats.to_csv(stats_path, index=False)

    # 요약 출력
    if not stats.empty:
        print("\n=== Summary (by event group & expiry) ===")
        print(stats.sort_values(["event","expiry_h"]).to_string(index=False))
    else:
        print("\n=== Summary === (no trades)")

    print(f"\n[BT] saved -> {trades_path} (rows={len(trades)})")
    print(f"[BT] saved -> {stats_path} (rows={len(stats)})")


if __name__ == "__main__":
    mp.freeze_support()
    main()