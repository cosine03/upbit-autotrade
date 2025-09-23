# -*- coding: utf-8 -*-
"""
TV signals 기반 Breakout Close-Entry 백테스트 (Single Process, 4h/8h 만기 동시 산출)

- Entry: 각 시그널의 "다음 봉 종가"로 롱 진입
- Exit: TP/SL 도달 또는 만기(4h/8h) 종가 청산
- Fees: 왕복 0.1% (roundtrip_fee=0.001)
- Scenarios: touches>=2 와 touches>=3 각각 집계
- Timeframe: 기본 15m (변경 가능)

사용 예:
  python backtest_tv_breakout_close_4h8h.py ./logs/signals_tv.csv --timeframe 15m --tp 1.5 --sl 1.0
"""

import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# OHLCV 로더 (sr_engine)
from sr_engine.data import get_ohlcv

# -------------------- 유틸: 타임존/시계열 통일 --------------------
def ensure_ohlcv_ts(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV ts를 UTC-naive datetime64[ns]로 통일하고 정렬"""
    if df is None or len(df) == 0:
        return None
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            ts = out.index
        else:
            raise ValueError("OHLCV에 ts 컬럼/DatetimeIndex가 없습니다.")
    else:
        ts = pd.to_datetime(out["ts"], utc=True, errors="coerce")

    # tz-aware -> UTC-naive
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    else:
        # tz-naive면 UTC 기준으로 본다(데이터 특성상 이미 UTC일 가능성 큼)
        ts = ts.tz_localize(None)

    out["ts"] = ts.astype("datetime64[ns]")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out


def series_to_ns_utc(s: pd.Series) -> np.ndarray:
    """datetime64[ns](tz-naive, UTC 가정) numpy 배열 반환"""
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]").to_numpy()


def parse_signal_ts_utc_naive(ts_value) -> pd.Timestamp:
    """시그널 ts를 UTC-aware로 파싱한 뒤 tz-naive(UTC)로 변환"""
    sig = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if sig is pd.NaT:
        return sig
    # tz-aware -> tz-naive(UTC)
    return sig.tz_convert("UTC").tz_localize(None)


def tf_minutes(tf: str) -> int:
    s = tf.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("d"):
        return int(s[:-1]) * 60 * 24
    return 15


def bars_for_hours(hours: float, timeframe: str) -> int:
    m = tf_minutes(timeframe)
    return max(1, int(round((hours * 60.0) / m)))


def idx_of_bar(ts64: np.ndarray, key64: np.datetime64) -> int:
    """key 시각이 속하는 봉 인덱스(오른쪽 경계 - 1). 범위 밖이면 {-1, len-1}."""
    i = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i < 0:
        return -1
    if i >= len(ts64):
        return len(ts64) - 1
    return i

# -------------------- 백테스트 로직 --------------------
def simulate_symbol(
    sym: str,
    ohlcv: pd.DataFrame,
    rows: pd.DataFrame,
    timeframe: str,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float,
    expiry_hours: float,
) -> pd.DataFrame:
    """
    각 시그널의 '다음 봉 종가'로 롱 진입 -> TP/SL/만기(4h/8h) 처리
    """
    if ohlcv is None or len(ohlcv) < 3 or rows is None or rows.empty:
        return pd.DataFrame()

    ts64 = series_to_ns_utc(ohlcv["ts"])
    closes = ohlcv["close"].to_numpy(dtype=float)
    highs  = ohlcv["high"].to_numpy(dtype=float)
    lows   = ohlcv["low"].to_numpy(dtype=float)

    hold_bars = bars_for_hours(expiry_hours, timeframe)
    trades = []

    for s in rows.itertuples():
        # 시그널 시간 -> UTC-naive ns
        sig_ts = parse_signal_ts_utc_naive(s.ts)
        if sig_ts is pd.NaT:
            continue
        sig64 = np.datetime64(sig_ts, "ns")

        i_sig = idx_of_bar(ts64, sig64)
        i_ent = i_sig + 1  # 다음 봉 종가 진입
        if i_ent >= len(ohlcv):
            continue

        entry_ts = ohlcv["ts"].iloc[i_ent]
        entry_px = float(closes[i_ent])

        tp_px = entry_px * (1.0 + tp_pct / 100.0)
        sl_px = entry_px * (1.0 - sl_pct / 100.0)

        i_end = min(len(ohlcv) - 1, i_ent + hold_bars)

        exit_idx = None
        exit_reason = "expiry"
        gross_ret = None

        # 다음 봉부터 만기까지 순회 (엔트리 봉 이후부터 체크)
        for j in range(i_ent + 1, i_end + 1):
            bar_low = float(lows[j])
            bar_high = float(highs[j])

            # 롱: SL 먼저 체크(보수적) -> TP
            if bar_low <= sl_px:
                exit_idx = j
                exit_reason = "SL"
                gross_ret = (sl_px - entry_px) / entry_px
                break
            if bar_high >= tp_px:
                exit_idx = j
                exit_reason = "TP"
                gross_ret = (tp_px - entry_px) / entry_px
                break

        if exit_idx is None:
            # 만기 종가 청산
            exit_idx = i_end
            exit_reason = "expiry"
            exit_px = float(closes[exit_idx])
            gross_ret = (exit_px - entry_px) / entry_px

        net_ret = gross_ret - fee_rt  # 왕복 수수료 차감
        trades.append({
            "symbol": sym,
            "signal_ts": pd.Timestamp(sig64),
            "entry_ts": pd.Timestamp(ohlcv["ts"].iloc[i_ent]),
            "entry_px": entry_px,
            "exit_ts": pd.Timestamp(ohlcv["ts"].iloc[exit_idx]),
            "exit_px": float(closes[exit_idx]) if exit_reason == "expiry" else (tp_px if exit_reason=="TP" else sl_px),
            "reason": exit_reason,
            "tp": tp_pct,
            "sl": sl_pct,
            "expiry_h": expiry_hours,
            "ret_gross": gross_ret,
            "ret_net": net_ret,
        })

    return pd.DataFrame(trades)


def agg_stats(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series({
            "trades": 0,
            "win_rate": np.nan,
            "avg_net": np.nan,
            "median_net": np.nan,
            "total_net": np.nan,
        })
    wins = (df["ret_net"] > 0).sum()
    return pd.Series({
        "trades": len(df),
        "win_rate": wins / len(df),
        "avg_net": df["ret_net"].mean(),
        "median_net": df["ret_net"].median(),
        "total_net": df["ret_net"].sum(),
    })


def run_touch_scenario(
    df_sig: pd.DataFrame,
    timeframe: str,
    tp: float,
    sl: float,
    fee_rt: float,
    expiry_hours_list: Tuple[float, float],
    touches_req: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    touches>=N 필터로 시나리오 실행(4h,8h 두 만기)
    """
    # 시그널 전처리: UTC-naive, 숫자형 변환
    df = df_sig.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["ts", "symbol"])
    df["touches"] = pd.to_numeric(df.get("touches", np.nan), errors="coerce")

    # 롱 전용: resistance만 사용 (폴 기준)
    df = df[df.get("side", "").astype(str).str.lower() == "resistance"].copy()

    # touches 필터
    df = df[df["touches"] >= touches_req].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    symbols = sorted(df["symbol"].dropna().unique().tolist())

    # 심볼별 OHLCV 로드/정규화
    ohlcv_map: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            ohl = get_ohlcv(sym, timeframe)
            ohl = ensure_ohlcv_ts(ohl)
            if ohl is None or ohl.empty or "close" not in ohl.columns:
                print(f"[WARN] OHLCV empty: {sym}")
                continue
            ohlcv_map[sym] = ohl
        except Exception as ex:
            print(f"[WARN] OHLCV load error for {sym}: {ex!r}")

    all_trades = []
    for sym in symbols:
        ohlcv = ohlcv_map.get(sym)
        if ohlcv is None:
            continue
        rows = df[df["symbol"] == sym]
        if rows.empty:
            continue
        for eh in expiry_hours_list:
            tr = simulate_symbol(sym, ohlcv, rows, timeframe, tp, sl, fee_rt, eh)
            if not tr.empty:
                tr["strategy"] = f"touch{touches_req}_{tp}/{sl}_{int(eh)}h"
                all_trades.append(tr)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats = trades.groupby("strategy", as_index=False, dropna=False).apply(agg_stats) if not trades.empty else pd.DataFrame()

    return trades, stats

# -------------------- 메인 --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("signals", type=str, help="TV signals CSV 경로 (signals_tv.csv)")
    p.add_argument("--timeframe", type=str, default="15m")
    p.add_argument("--tp", type=float, default=1.5, help="take-profit %")
    p.add_argument("--sl", type=float, default=1.0, help="stop-loss %")
    p.add_argument("--fee", type=float, default=0.001, help="roundtrip fee (e.g. 0.001 = 0.1%)")
    args = p.parse_args()

    df_sig = pd.read_csv(args.signals)
    print(f"[BT] signals rows={len(df_sig)}, symbols={df_sig['symbol'].nunique()}, timeframe={args.timeframe}")

    expiry_list = (4.0, 8.0)

    # touches >= 2
    t2_trades, t2_stats = run_touch_scenario(
        df_sig, args.timeframe, args.tp, args.sl, args.fee, expiry_list, touches_req=2
    )
    if not t2_stats.empty:
        print("\n=== touches>=2 ===")
        print(t2_stats)

    # touches >= 3
    t3_trades, t3_stats = run_touch_scenario(
        df_sig, args.timeframe, args.tp, args.sl, args.fee, expiry_list, touches_req=3
    )
    if not t3_stats.empty:
        print("\n=== touches>=3 ===")
        print(t3_stats)

    # 저장
    os.makedirs("./logs", exist_ok=True)
    out_trades = "./logs/bt_tv_breakout_close_4h8h_trades.csv"
    out_stats  = "./logs/bt_tv_breakout_close_4h8h_stats.csv"

    trades_all = pd.concat(
        [x for x in [t2_trades, t3_trades] if x is not None and not x.empty],
        ignore_index=True
    ) if (t2_trades is not None or t3_trades is not None) else pd.DataFrame()

    stats_all = pd.concat(
        [x for x in [t2_stats, t3_stats] if x is not None and not x.empty],
        ignore_index=True
    ) if (t2_stats is not None or t3_stats is not None) else pd.DataFrame()

    if not trades_all.empty:
        trades_all.to_csv(out_trades, index=False)
    if not stats_all.empty:
        stats_all.to_csv(out_stats, index=False)

    print("\n[BT] 저장 완료.")
    if not trades_all.empty:
        print(f"  → {out_trades} (rows={len(trades_all)})")
    else:
        print(f"  → {out_trades} (no trades)")
    if not stats_all.empty:
        print(f"  → {out_stats} (rows={len(stats_all)})")
    else:
        print(f"  → {out_stats} (no stats)")

if __name__ == "__main__":
    main()