# -*- coding: utf-8 -*-
"""
backtest_tv_events_mp.py  (TV 알람 4종 멀티프로세싱 안정판)

- 이벤트 그룹: detected / price_in_box / box_breakout / line_breakout
- 엔트리: 알람 봉 다음 봉 시가(롱 온리), TP/SL 퍼센트, 만기(4h,8h 등) 체크
- 수수료: 기본 0.1% (왕복) -> --fee 0.001
- 멀티프로세싱: 심볼 단위로 병렬 (pickle-safe payload만 전달)
- UTC/타임존: 모든 타임스탬프 UTC-naive ns 로 통일 (tz 충돌 방지)
- 빈 OHLCV: 안전하게 스킵 (로그만 남김)
- 그룹 통계: groupby.agg 로 FutureWarning 회피

사용 예)
  python backtest_tv_events_mp.py ./logs/signals_tv.csv `
    --timeframe 15m `
    --expiries 4h,8h `
    --tp 1.5 --sl 1.25 --fee 0.001 `
    --procs 24

출력:
  ./logs/bt_tv_events_trades_<group>.csv
  ./logs/bt_tv_events_stats_<group>.csv
  요약 합본:
  ./logs/bt_tv_events_trades.csv
  ./logs/bt_tv_events_stats.csv
"""

import os
import sys
import csv
import math
import time
import argparse
import traceback
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

# 멀티프로세싱
import multiprocessing as mp

# 프로젝트 OHLCV 로더 (pyupbit 래핑되어 있을 것)
from sr_engine.data import get_ohlcv


# ---------------------------- 유틸: 경로/출력 ----------------------------
def ensure_logs_dir(path: str) -> str:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    return path


# ---------------------------- UTC / 타임존 정규화 ----------------------------
def to_utc_naive_ts(s: pd.Series) -> pd.Series:
    """Series → UTC-naive pandas.Timestamp Series (ns)"""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    # 허용 입력: str / pandas.Timestamp / numpy datetime64 / DatetimeTZ
    s = pd.to_datetime(s, utc=True, errors="coerce")
    # tz-aware(UTC) -> tz-naive
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def series_to_ns_utc(s: pd.Series) -> np.ndarray:
    """UTC-naive pandas.Timestamp Series -> numpy datetime64[ns]"""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    # s 가 이미 tz-naive pandas.Timestamp 라고 가정
    # pandas >=2.2: .astype(datetime64[ns])는 tz-aware에 대해 TypeError 유발
    # 여기선 이미 tz-naive이므로 안전
    return s.astype("datetime64[ns]").to_numpy()


def ts_to_ns64(ts: pd.Timestamp) -> np.datetime64:
    """개별 Timestamp -> numpy datetime64[ns] (UTC-naive 기준)"""
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.to_datetime64())


# ---------------------------- OHLCV 로드 ----------------------------
def load_ohlcv(symbol: str, timeframe: str, retry: int = 2, sleep_s: float = 0.7) -> pd.DataFrame:
    """
    get_ohlcv(symbol, timeframe) 결과를 표준 포맷으로 정리
    - 컬럼: ts(open/high/low/close/volume)
    - ts: UTC-naive pandas.Timestamp
    - 실패/빈값이면 empty DF 반환
    """
    last_ex = None
    for _ in range(max(1, retry)):
        try:
            df = get_ohlcv(symbol, timeframe)
            if df is None or len(df) == 0:
                last_ex = None
            else:
                # 인덱스가 DatetimeIndex 라는 전제 하에 ts 칼럼 생성
                if isinstance(df.index, pd.DatetimeIndex):
                    ts = df.index
                else:
                    # 혹시 모를 변형
                    ts = pd.to_datetime(df.get("ts") or df.get("time") or df.index, errors="coerce", utc=True)
                ts = ts.tz_convert("UTC").tz_localize(None)
                out = pd.DataFrame({
                    "ts": ts,
                    "open": pd.to_numeric(df["open"], errors="coerce"),
                    "high": pd.to_numeric(df["high"], errors="coerce"),
                    "low":  pd.to_numeric(df["low"],  errors="coerce"),
                    "close":pd.to_numeric(df["close"],errors="coerce"),
                    "volume":pd.to_numeric(df.get("volume", np.nan), errors="coerce"),
                })
                out = out.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)
                if not out.empty:
                    return out
        except Exception as ex:
            last_ex = ex
        time.sleep(sleep_s)

    print(f"[{symbol}] get_ohlcv returned empty." + (f" last_ex={repr(last_ex)}" if last_ex else ""))
    return pd.DataFrame(columns=["ts","open","high","low","close","volume"])


# ---------------------------- 시뮬레이션 로직 ----------------------------
def next_bar_open_idx(ts64: np.ndarray, sig_ts: pd.Timestamp) -> int:
    """
    알람 봉의 '다음 봉 시가' 인덱스를 찾는다.
    ts64: numpy datetime64[ns] (정렬 가정)
    sig_ts: pandas.Timestamp (임의 tz) -> UTC-naive 로 변환 후 searchsorted
    """
    if not isinstance(sig_ts, pd.Timestamp):
        sig_ts = pd.Timestamp(sig_ts)

    if sig_ts.tzinfo is not None:
        sig_ts = sig_ts.tz_convert("UTC").tz_localize(None)

    key64 = np.datetime64(sig_ts.to_datetime64())
    # right -> sig_ts 와 같은 시각이면 다음 위치
    idx = int(np.searchsorted(ts64, key64, side="right"))
    return idx


def simulate_one(
    ohlcv: pd.DataFrame,
    sig_ts: pd.Timestamp,
    side: str,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float,
    expiry_h: float,
) -> Dict[str, Any]:
    """
    알람 1건에 대한 트레이드 시뮬
    - 엔트리: 알람 봉 '다음 봉'의 시가
    - TP/SL: 퍼센트 (예: TP 1.5% -> 0.015)
    - 수수료: 왕복 fee_rt (예: 0.001 = 0.1%)
    - 만기: expiry_h 시간
    """
    if ohlcv is None or ohlcv.empty:
        return {}

    # 타임축 준비
    ts = to_utc_naive_ts(ohlcv["ts"])
    ts64 = series_to_ns_utc(ts)

    # next open idx
    i_entry = next_bar_open_idx(ts64, sig_ts)
    if i_entry <= 0 or i_entry >= len(ohlcv):
        return {}

    entry_ts = ts.iloc[i_entry]
    entry_px = float(ohlcv["open"].iloc[i_entry])

    # 롱 온리
    tp_px = entry_px * (1.0 + tp_pct / 100.0)
    sl_px = entry_px * (1.0 - sl_pct / 100.0)

    # 만기 시각
    dt_expiry = entry_ts + pd.Timedelta(hours=expiry_h)
    exp64 = np.datetime64(dt_expiry.to_datetime64())
    i_exp = int(np.searchsorted(ts64, exp64, side="left"))
    if i_exp <= i_entry:
        i_exp = i_entry + 1
    i_end = min(i_exp, len(ohlcv) - 1)

    exit_ts = ts.iloc[i_end]
    exit_px = float(ohlcv["close"].iloc[i_end])
    hit = "expiry"

    # 엔트리 이후 구간에서 TP/SL 충족 여부 체크
    # (한 봉 내 TP/SL 동시히트 순서 문제 단순화: 먼저 TP 가격 이상이면 TP 우선, 아니면 SL)
    highs = ohlcv["high"].to_numpy()
    lows  = ohlcv["low"].to_numpy()
    for i in range(i_entry + 1, i_end + 1):
        if highs[i] >= tp_px:
            exit_ts = ts.iloc[i]
            exit_px = tp_px
            hit = "tp"
            break
        if lows[i] <= sl_px:
            exit_ts = ts.iloc[i]
            exit_px = sl_px
            hit = "sl"
            break

    gross = (exit_px / entry_px) - 1.0  # 수익률
    net = gross - fee_rt  # 왕복 수수료 차감(단순화)

    return {
        "entry_ts": entry_ts,
        "entry_px": entry_px,
        "exit_ts": exit_ts,
        "exit_px": exit_px,
        "gross": gross,
        "net": net,
        "hit": hit,
    }


# ---------------------------- 워커: 심볼 단위 ----------------------------
def simulate_symbol(
    symbol: str,
    timeframe: str,
    rows_payload: List[Dict[str, Any]],
    tp: float,
    sl: float,
    fee_rt: float,
    expiries_h: Tuple[float, ...],
    group_name: str,
) -> pd.DataFrame:
    """
    멀티프로세싱 워커.
    - rows_payload: dict record 리스트 (pickle-safe)
      필요한 키: ts(str/ts), side(str), event(str), level(int/str), touches(int/str)
    """
    try:
        ohlcv = load_ohlcv(symbol, timeframe)
        if ohlcv is None or ohlcv.empty:
            return pd.DataFrame()

        trades: List[Dict[str, Any]] = []
        for rec in rows_payload:
            # sig_ts UTC-naive 강제
            sig_ts = pd.to_datetime(rec.get("ts"), utc=True, errors="coerce")
            sig_ts = sig_ts.tz_convert("UTC").tz_localize(None)

            side = str(rec.get("side", "resistance")).strip().lower()
            level = rec.get("level", "")
            touches = rec.get("touches", "")
            event = rec.get("event", "")

            for eh in expiries_h:
                one = simulate_one(
                    ohlcv=ohlcv,
                    sig_ts=sig_ts,
                    side=side,
                    tp_pct=tp,
                    sl_pct=sl,
                    fee_rt=fee_rt,
                    expiry_h=float(eh),
                )
                if not one:
                    continue
                one.update({
                    "symbol": symbol,
                    "side": side,
                    "level": level,
                    "touches": touches,
                    "event": event,
                    "group": group_name,
                    "expiry_h": float(eh),
                    "tp": tp,
                    "sl": sl,
                    "fee": fee_rt,
                })
                trades.append(one)

        return pd.DataFrame(trades)

    except Exception as ex:
        print(f"[{symbol}] worker error: {repr(ex)}")
        traceback.print_exc()
        return pd.DataFrame()


# ---------------------------- 집계 ----------------------------
def aggregate_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """
    그룹별 요약 (FutureWarning 회피: groupby.agg 사용)
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["group","expiry_h","trades","win_rate","avg_net","median_net","total_net"])

    gb = trades.groupby(["group", "expiry_h"], dropna=False, as_index=False)
    out = gb.agg(
        trades=("net", "count"),
        win_rate=("net", lambda s: float((s > 0).mean()) if len(s) else 0.0),
        avg_net=("net", "mean"),
        median_net=("net", "median"),
        total_net=("net", "sum"),
    )
    return out


# ---------------------------- 메인 ----------------------------
def main():
    ap = argparse.ArgumentParser(description="TV 이벤트 4종 멀티프로세싱 백테스트 (안정판)")
    ap.add_argument("signals", help="TV signals csv 경로")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h", help="예: 4h,8h,12h")
    ap.add_argument("--tp", type=float, default=1.5, help="TP %, 예: 1.5")
    ap.add_argument("--sl", type=float, default=1.25, help="SL %, 예: 1.25 (요청 반영)")
    ap.add_argument("--fee", type=float, default=0.001, help="왕복 수수료 비율 (0.1% = 0.001)")
    ap.add_argument("--procs", type=int, default=min(8, (os.cpu_count() or 8)), help="프로세스 수")
    ap.add_argument("--outdir", default="./logs")
    args = ap.parse_args()

    expiries_h: Tuple[float, ...] = tuple(
        float(x.strip().lower().replace("h","")) for x in args.expiries.split(",") if x.strip()
    )
    os.makedirs(args.outdir, exist_ok=True)

    # 1) 신호 로드
    df = pd.read_csv(args.signals)
    # 표준 컬럼 보정
    for c in ["event","side","symbol","ts"]:
        if c not in df.columns:
            df[c] = ""

    # TV만 사용 (source가 있으면 필터)
    if "source" in df.columns:
        df = df[df["source"].astype(str).str.upper().str.contains("TV")]

    # ts 정규화 (UTC-naive)
    df["ts"] = to_utc_naive_ts(df["ts"])

    # 심볼/이벤트/터치/레벨 정리
    df["event"] = df["event"].astype(str).str.strip()
    df["side"] = df["side"].astype(str).str.strip().str.lower()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    if "touches" in df.columns:
        df["touches"] = pd.to_numeric(df["touches"], errors="coerce")
    else:
        df["touches"] = np.nan
    if "level" in df.columns:
        df["level"] = pd.to_numeric(df["level"], errors="coerce")
    else:
        df["level"] = np.nan

    # 사용 이벤트 4종
    event_order = ["detected", "price_in_box", "box_breakout", "line_breakout"]
    # message 등이 깨끗하지 않아도 event 열 값 기준으로 필터링
    used = df[df["event"].isin(event_order)].copy()
    used = used.sort_values(["symbol","ts"]).reset_index(drop=True)

    symbols = sorted(used["symbol"].dropna().unique().tolist())
    print(f"[BT] signals rows={len(used)}, symbols={len(symbols)}, timeframe={args.timeframe}")

    summary_rows = []
    all_trades_concat = []

    # 이벤트 그룹별 처리
    for grp in event_order:
        grp_df = used[used["event"] == grp].copy()
        if grp in ("detected", "price_in_box"):
            # Paul 문서 기준 정확도 낮음 -> touches >= 2 필터 기본 적용 (가능한 경향)
            # 단, signals에 이미 level/touches가 안정적이지 않다면 완화 필요
            # 여기선 없는 경우도 고려하여 NaN은 허용(필터 미적용)
            pass

        rows_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for sym, srows in grp_df.groupby("symbol"):
            payload = srows[["ts","side","event","level","touches"]].to_dict("records")
            rows_by_symbol[sym] = payload

        print(f"[BT][{grp}] start: symbols={len(rows_by_symbol)} rows={len(grp_df)} tasks={len(rows_by_symbol)} procs={args.procs}")

        # 멀티프로세싱 실행
        tasks = [
            (sym, args.timeframe, rows_by_symbol[sym], args.tp, args.sl, args.fee, expiries_h, grp)
            for sym in rows_by_symbol.keys()
        ]

        if not tasks:
            print(f"[BT][{grp}] no tasks.")
            continue

        # Windows 안전: spawn
        # (이미 기본 spawn 이지만, 명시해도 OK)
        mp.freeze_support()
        with mp.Pool(processes=args.procs) as pool:
            parts = pool.starmap(simulate_symbol, tasks)

        # 합치기
        trades = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
        trades_out = ensure_logs_dir(os.path.join(args.outdir, f"bt_tv_events_trades_{grp}.csv"))
        stats_out  = ensure_logs_dir(os.path.join(args.outdir, f"bt_tv_events_stats_{grp}.csv"))

        if not trades.empty:
            trades.to_csv(trades_out, index=False)
            stats = aggregate_stats(trades)
            stats.to_csv(stats_out, index=False)
            print(f"[BT][{grp}] trades -> {trades_out} (rows={len(trades)})")
            print(f"[BT][{grp}] stats  -> {stats_out} (rows={len(stats)})")

            all_trades_concat.append(trades)
            # 요약행 추가
            for _, r in stats.iterrows():
                summary_rows.append({
                    "group": grp,
                    "expiry_h": float(r["expiry_h"]),
                    "trades": int(r["trades"]),
                    "win_rate": float(r["win_rate"]),
                    "avg_net": float(r["avg_net"]),
                    "median_net": float(r["median_net"]),
                    "total_net": float(r["total_net"]),
                })
        else:
            # 빈 경우에도 파일은 남겨두면 관리가 편함
            pd.DataFrame(columns=[
                "symbol","side","level","touches","event","group","expiry_h",
                "tp","sl","fee","entry_ts","entry_px","exit_ts","exit_px","gross","net","hit"
            ]).to_csv(trades_out, index=False)
            pd.DataFrame(columns=["group","expiry_h","trades","win_rate","avg_net","median_net","total_net"]).to_csv(stats_out, index=False)
            print(f"[BT][{grp}] no trades. (wrote empty files)")

    # 전체 요약
    if all_trades_concat:
        all_trades = pd.concat(all_trades_concat, ignore_index=True)
        all_stats  = aggregate_stats(all_trades)

        all_trades_out = ensure_logs_dir(os.path.join(args.outdir, "bt_tv_events_trades.csv"))
        all_stats_out  = ensure_logs_dir(os.path.join(args.outdir, "bt_tv_events_stats.csv"))
        all_trades.to_csv(all_trades_out, index=False)
        all_stats.to_csv(all_stats_out, index=False)

        print("\n=== Summary (by event group & expiry) ===")
        print(all_stats.to_string(index=False))
        print(f"\n[BT] saved -> {all_trades_out} (rows={len(all_trades)})")
        print(f"[BT] saved -> {all_stats_out} (rows={len(all_stats)})")
    else:
        print("\n[BT] no trades at all.")


if __name__ == "__main__":
    main()