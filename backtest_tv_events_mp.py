# -*- coding: utf-8 -*-
"""
backtest_tv_events_mp.py  (TV 알람 4종 멀티프로세싱 백테스트)
- 엔트리 정확도 보강: distance_pct 필터 (--dist-max, 기본 0.25%)
- TP 고정 1.5% / SL 상향 1.25% (기본값) / 수수료 roundtrip 기본 0.1% (=0.001)
- 만기 4h, 8h 동시 평가
- 빈 OHLCV 대응: 재시도 + 캐시(csv) + UTC 정규화
- 멀티프로세싱 기본 24프로세스

사용 예:
  python backtest_tv_events_mp.py .\logs\signals_tv.csv --timeframe 15m --expiries 4h,8h --tp 1.5 --sl 1.25 --fee 0.001 --dist-max 0.25 --procs 24
"""
import os, sys, argparse, time, hashlib, itertools
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# === 프로젝트 내 최소 의존: get_ohlcv ===
from sr_engine.data import get_ohlcv

# --------------------- 공통 유틸 ---------------------
def tf_minutes(tf: str) -> int:
    s = tf.strip().lower()
    if s.endswith("m"): return int(s[:-1])
    if s.endswith("h"): return int(s[:-1]) * 60
    if s.endswith("d"): return int(s[:-1]) * 60 * 24
    return 15

def safe_to_utc_series(s: pd.Series) -> pd.Series:
    """Series를 UTC-aware로 통일."""
    if pd.api.types.is_datetime64_any_dtype(s):
        # tz-aware?
        if getattr(s.dt, 'tz', None) is not None:
            return s.dt.tz_convert("UTC")
        return s.dt.tz_localize("UTC")
    # 문자열/숫자 등
    return pd.to_datetime(s, utc=True, errors="coerce")

def series_to_ns_utc(s: pd.Series) -> np.ndarray:
    """UTC-aware 시계를 int64(ns) ndarray로."""
    s = safe_to_utc_series(s)
    # tz-aware → ns
    return s.view("int64").to_numpy()

def parse_sig_ts(x) -> pd.Timestamp:
    """신호 ts 파싱 → UTC-aware Timestamp"""
    return pd.to_datetime(x, utc=True, errors="coerce")

def expiry_to_hours_list(s: str) -> List[float]:
    out = []
    for part in s.split(","):
        part = part.strip().lower()
        if not part: continue
        if part.endswith("h"):
            out.append(float(part[:-1]))
        elif part.endswith("d"):
            out.append(float(part[:-1]) * 24.0)
        else:
            out.append(float(part))  # 시간 단위로 간주
    return out

def cache_path(symbol: str, timeframe: str) -> str:
    h = hashlib.md5(f"{symbol}|{timeframe}".encode()).hexdigest()[:8]
    os.makedirs("./cache_ohlcv", exist_ok=True)
    return f"./cache_ohlcv/{symbol.replace('/','_')}_{timeframe}_{h}.csv"

@lru_cache(maxsize=256)
def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    - 캐시(csv) → API(재시도) → 빈 DF
    - 컬럼: ts, open, high, low, close
    - ts: UTC-aware
    """
    cp = cache_path(symbol, timeframe)
    # 1) 캐시
    if os.path.exists(cp):
        try:
            dfc = pd.read_csv(cp)
            if "ts" in dfc:
                dfc["ts"] = safe_to_utc_series(dfc["ts"])
            need = {"ts", "open", "high", "low", "close"}
            if need.issubset(dfc.columns):
                dfc = (dfc[list(need)]
                       .dropna(subset=list(need))
                       .sort_values("ts")
                       .reset_index(drop=True))
                if not dfc.empty:
                    return dfc
        except Exception:
            pass

    # 2) API 재시도
    for wait in (0.7, 1.4, 2.0, 3.0):
        df = get_ohlcv(symbol, timeframe)
        if df is not None and len(df) > 0:
            # 인덱스/컬럼 상황에 따라 ts 생성
            if isinstance(df.index, pd.DatetimeIndex):
                ts = df.index
                ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
                d2 = df.copy()
                d2["ts"] = ts
                d2 = d2.reset_index(drop=True)
            else:
                d2 = df.reset_index(drop=False)
                # ts 후보 찾기
                ts_col = None
                for c in d2.columns:
                    if pd.api.types.is_datetime64_any_dtype(d2[c]):
                        ts_col = c; break
                if ts_col is None:
                    for cand in ("ts","index","timestamp","time","datetime","date"):
                        if cand in d2.columns:
                            parsed = pd.to_datetime(d2[cand], utc=True, errors="coerce")
                            if parsed.notna().any():
                                d2["ts"] = parsed; ts_col = "ts"; break
                if ts_col is None and "ts" not in d2.columns:
                    print(f"[{symbol}] OHLCV missing datetime column; skipping.")
                    return pd.DataFrame()
                if "ts" not in d2.columns:
                    d2 = d2.rename(columns={ts_col:"ts"})
                    d2["ts"] = pd.to_datetime(d2["ts"], utc=True, errors="coerce")

            need = ["ts","open","high","low","close"]
            d2 = (d2[need]
                  .dropna(subset=need)
                  .sort_values("ts")
                  .reset_index(drop=True))
            try:
                d2.to_csv(cp, index=False)
            except Exception:
                pass
            return d2
        time.sleep(wait)

    print(f"[{symbol}] get_ohlcv returned empty.")
    return pd.DataFrame()

def idx_of_bar(ts64: np.ndarray, key_ts: pd.Timestamp) -> int:
    """
    시그널이 발생한 봉의 인덱스(닫힌 봉 기준)를 반환.
    ts64: ns int64 배열(UTC)
    key_ts: UTC-aware Timestamp
    """
    key64 = np.datetime64(key_ts.to_datetime64())
    idx = int(np.searchsorted(ts64, key64, side="right")) - 1
    return max(0, idx)

def bars_until(ts64: np.ndarray, start_i: int, hours: float, tf_min: int) -> int:
    """start_i부터 hours가 지나는 인덱스(만기 직전까지)를 대략 계산."""
    bars = int(np.ceil((hours * 60) / tf_min))
    return min(len(ts64) - 1, start_i + bars)

def simulate_symbol(
    symbol: str,
    ohlcv: pd.DataFrame,
    rows: pd.DataFrame,
    timeframe: str,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float,
    exp_hours: float
) -> pd.DataFrame:
    """
    시그널 단위로 다음 봉 오픈에 진입(롱), TP/SL/만기으로 청산.
    동시히트 시 SL 우선(보수적).
    """
    if ohlcv is None or ohlcv.empty or rows.empty:
        return pd.DataFrame()

    ts64 = series_to_ns_utc(ohlcv["ts"])
    open_ = ohlcv["open"].to_numpy(dtype=float, copy=False)
    high_ = ohlcv["high"].to_numpy(dtype=float, copy=False)
    low__ = ohlcv["low"].to_numpy(dtype=float, copy=False)
    close = ohlcv["close"].to_numpy(dtype=float, copy=False)
    tf_min = tf_minutes(timeframe)

    out = []
    for s in rows.itertuples(index=False):
        sig_ts = parse_sig_ts(getattr(s, "ts", None))
        side = getattr(s, "side", "support")
        event = getattr(s, "event", "")
        if pd.isna(sig_ts):
            continue

        i_sig = idx_of_bar(ts64, sig_ts)
        i_ent = i_sig + 1
        if i_ent >= len(ohlcv):
            continue

        # 롱 온리
        px_e = float(open_[i_ent])
        tp_px = px_e * (1.0 + tp_pct)
        sl_px = px_e * (1.0 - sl_pct)

        # 만기 인덱스
        i_last = bars_until(ts64, i_ent, exp_hours, tf_min)

        exit_px = float(close[i_last])
        exit_ts = pd.to_datetime(ts64[i_last]).tz_localize("UTC")
        reason = "expiry"

        # 각 봉에서 TP/SL 충돌 체크 (SL 먼저)
        for i in range(i_ent, i_last + 1):
            lo = float(low__[i]); hi = float(high_[i])
            if lo <= sl_px:
                exit_px = sl_px
                exit_ts = pd.to_datetime(ts64[i]).tz_localize("UTC")
                reason = "SL"
                break
            if hi >= tp_px:
                exit_px = tp_px
                exit_ts = pd.to_datetime(ts64[i]).tz_localize("UTC")
                reason = "TP"
                break

        # 수수료(왕복): fee_rt
        gross = (exit_px / px_e) - 1.0
        net = gross - fee_rt

        out.append({
            "symbol": symbol,
            "event": event,
            "side": side,
            "entry_ts": pd.to_datetime(ts64[i_ent]).tz_localize("UTC").isoformat(),
            "entry_px": px_e,
            "exit_ts": exit_ts.isoformat(),
            "exit_px": exit_px,
            "reason": reason,
            "tp": round(tp_pct*100, 3),
            "sl": round(sl_pct*100, 3),
            "expiry_h": float(exp_hours),
            "gross": gross,
            "net": net,
        })

    return pd.DataFrame(out)

def agg_stats(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series({"trades": 0, "win_rate": np.nan, "avg_net": np.nan, "median_net": np.nan, "total_net": 0.0})
    wins = (df["net"] > 0).mean() if len(df) else 0.0
    return pd.Series({
        "trades": float(len(df)),
        "win_rate": float(wins),
        "avg_net": float(df["net"].mean()),
        "median_net": float(df["net"].median()),
        "total_net": float(df["net"].sum()),
    })

# --------------------- 메인 로직 ---------------------
EVENT_GROUPS = {
    "detected": {"level2_detected", "level3_detected"},
    "price_in_box": {"price_in_box"},
    "box_breakout": {"box_breakout"},
    "line_breakout": {"line_breakout"},
}

def run_group(
    df_sig: pd.DataFrame,
    group: str,
    timeframe: str,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float,
    expiries_h: List[float],
    procs: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from multiprocessing import get_context
    symbols = sorted(df_sig["symbol"].unique().tolist())
    tasks = []
    for sym in symbols:
        ohlcv = load_ohlcv(sym, timeframe)  # 캐시+재시도
        if ohlcv is None or ohlcv.empty:
            continue
        rows = df_sig[df_sig["symbol"] == sym].copy()
        if rows.empty:
            continue
        for eh in expiries_h:
            tasks.append((sym, ohlcv, rows, timeframe, tp_pct, sl_pct, fee_rt, float(eh)))

    print(f"[BT][{group}] start: symbols={len(symbols)} rows={len(df_sig)} tasks={len(tasks)} procs={procs}")

    parts: List[pd.DataFrame] = []
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    with get_context("spawn").Pool(processes=procs) as pool:
        parts = pool.starmap(simulate_symbol, tasks)

    trades = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    stats = (trades.groupby(["expiry_h"], as_index=False).apply(agg_stats)
             if not trades.empty else pd.DataFrame())
    return trades, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv 경로(또는 enriched csv)")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h", help="예: 4h,8h")
    ap.add_argument("--tp", type=float, default=1.5, help="TP % (예: 1.5)")
    ap.add_argument("--sl", type=float, default=1.25, help="SL % (예: 1.25)")
    ap.add_argument("--fee", type=float, default=0.001, help="왕복 수수료 비율 (기본 0.001=0.1%)")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--dist-max", type=float, default=0.25, help="distance_pct 필터 (절대값 %) — 열 없으면 미적용")
    args = ap.parse_args()

    expiries_h = expiry_to_hours_list(args.expiries)
    tp_pct = args.tp / 100.0
    sl_pct = args.sl / 100.0

    df = pd.read_csv(args.signals)
    # 표준화
    need_cols = ["ts","event","side","symbol"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = ""
    df["ts"] = parse_sig_ts(df["ts"])
    df = df.dropna(subset=["ts","event","side","symbol"]).reset_index(drop=True)

    # TV만 사용(가능하면)
    if "source" in df.columns:
        df = df[df["source"].astype(str).str.upper().str.contains("TV") | df["source"].isna()].reset_index(drop=True)

    # distance_pct 필터 (열이 있을 때만 적용)
    if "distance_pct" in df.columns:
        df["distance_pct"] = pd.to_numeric(df["distance_pct"], errors="coerce")
        before = len(df)
        df = df[df["distance_pct"].abs() <= args.dist_max].reset_index(drop=True)
        print(f"[BT] distance_pct ≤ {args.dist_max:.4f}% 필터: {before} → {len(df)} rows")

    # 이벤트 그룹별로 분리
    rows_by_group: Dict[str, pd.DataFrame] = {}
    for g, names in EVENT_GROUPS.items():
        rows_by_group[g] = df[df["event"].isin(names)].copy()

    syms = sorted(df["symbol"].unique().tolist())
    print(f"[BT] signals rows={len(df)}, symbols={len(syms)}, timeframe={args.timeframe}")

    all_trades = []
    all_stats = []
    os.makedirs("./logs", exist_ok=True)

    for grp in ("detected","price_in_box","box_breakout","line_breakout"):
        dfg = rows_by_group.get(grp, pd.DataFrame())
        if dfg.empty:
            print(f"[BT][{grp}] no rows, skip.")
            continue
        tr, st = run_group(dfg, grp, args.timeframe, tp_pct, sl_pct, args.fee, expiries_h, args.procs)
        # 저장
        tpath = f"./logs/bt_tv_events_trades_{grp}.csv"
        spath = f"./logs/bt_tv_events_stats_{grp}.csv"
        tr.to_csv(tpath, index=False)
        st.insert(0, "group", grp)
        st.to_csv(spath, index=False)
        print(f"[BT][{grp}] trades -> {tpath} (rows={len(tr)})")
        print(f"[BT][{grp}] stats  -> {spath} (rows={len(st)})")

        all_trades.append(tr.assign(group=grp))
        all_stats.append(st)

    # 요약
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    if not stats.empty:
        stats = stats[["group","expiry_h","trades","win_rate","avg_net","median_net","total_net"]].copy()
        print("\n=== Summary (by event group & expiry) ===")
        print(stats.to_string(index=False))

    trades.to_csv("./logs/bt_tv_events_trades.csv", index=False)
    stats.to_csv("./logs/bt_tv_events_stats.csv", index=False)
    print(f"\n[BT] saved -> ./logs/bt_tv_events_trades.csv (rows={len(trades)})")
    print(f"[BT] saved -> ./logs/bt_tv_events_stats.csv (rows={len(stats)})")

if __name__ == "__main__":
    main()