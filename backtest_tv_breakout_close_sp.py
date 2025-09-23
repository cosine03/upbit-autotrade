# -*- coding: utf-8 -*-
"""
backtest_tv_breakout_close_sp.py
- TV signals 기반 (price_in_box / levelX_detected 등) 백테스트
- 엔트리: "신호 봉의 다음 봉 종가"로 진입(= close-breakout 가정의 간단 버전)
- TP/SL: 퍼센트, 라운드트립 수수료(기본 0.1%) 반영
- 만기: N시간(expiry_h), 만기까지 TP/SL 미체결 시 만기 시점 종가 청산
- tz-naive/aware 충돌 방지: 모든 시계열을 'UTC-naive' 로 통일
- OHLCV 로딩 실패는 안전하게 스킵(전체 진행 중단 방지)

사용 예:
  python backtest_tv_breakout_close_sp.py ./logs/signals_tv.csv ^
    --timeframe 15m --expiry 24h --tp 1.5 --sl 1.0 --touches 2 --touches-alt 3
"""

import argparse
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ---- pyupbit 로딩 (필요 시 설치 안내) ----
try:
    import pyupbit  # type: ignore
except Exception as e:
    raise SystemExit(
        "pyupbit 모듈을 찾을 수 없습니다. 가상환경에서 아래를 먼저 실행하세요.\n"
        "  pip install pyupbit\n원인: " + repr(e)
    )

# -------------------- 공통 유틸 --------------------
def to_utc_naive(ts_like) -> pd.Timestamp:
    """입력(ts/str/np.datetime64 등)을 UTC naive Timestamp로 변환."""
    t = pd.Timestamp(ts_like)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    else:
        # tz 없는 경우도 'UTC 기준'으로 간주(naive)
        t = t.tz_localize(None)
    return t


def series_to_utc_naive(s: pd.Series) -> pd.Series:
    """시리즈를 UTC naive로 변환."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    # tz-aware -> tz-naive
    t = t.dt.tz_convert(None)
    return t


def timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 15


def parse_expiry_hours(s: str) -> float:
    """'24h' 같은 문자열을 시간(float)으로 변환"""
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1])
    return float(s)


# -------------------- OHLCV --------------------
def get_ohlcv_safe(symbol: str, timeframe: str, count: int = 2000,
                   retries: int = 3, pause: float = 0.5) -> Optional[pd.DataFrame]:
    """
    pyupbit.get_ohlcv 안전 래퍼.
    - 실패 시 재시도
    - 성공 시 컬럼: ['ts','open','high','low','close','volume','value'(있으면)]
      ts는 UTC-naive의 pandas.Timestamp
    """
    iv = timeframe
    last_err = None
    for r in range(retries):
        try:
            df = pyupbit.get_ohlcv(symbol, interval=iv, count=count)
            if df is None or len(df) == 0:
                raise RuntimeError("empty dataframe returned")

            df = df.reset_index()
            # pyupbit index -> 'index' 컬럼이 Timestamp (tz-naive 또는 tz-aware)
            # 모두 UTC naive로 정규화
            df["ts"] = series_to_utc_naive(df["index"])
            df = df.drop(columns=["index"])
            # 정렬/결측 제거
            df = df.sort_values("ts").dropna(subset=["ts"]).reset_index(drop=True)
            # 기본 컬럼 보정
            need = ["open", "high", "low", "close", "volume"]
            for c in need:
                if c not in df.columns:
                    raise RuntimeError(f"missing column: {c}")
            return df[["ts","open","high","low","close","volume"] + ([c for c in df.columns if c not in ["ts","open","high","low","close","volume"]])]
        except Exception as e:
            last_err = e
            print(f"[{symbol}] OHLCV load error: {e}")
            time.sleep(pause)
    print(f"[{symbol}] OHLCV failed after retries. skip.")
    return None


# -------------------- 인덱스/만기 계산 --------------------
def idx_of_bar(ts: pd.Series, sig_ts: pd.Timestamp) -> int:
    """
    시그널이 발생한 시각(sig_ts)이 소속된 봉 index (오른쪽-닫힘 기준 right) - 1
    모든 ts와 sig_ts는 UTC-naive로 통일되어야 한다.
    """
    ts64 = ts.astype("datetime64[ns]")  # tz-naive ndarray
    key = np.datetime64(to_utc_naive(sig_ts))
    idx = int(np.searchsorted(ts64, key, side="right")) - 1
    return max(idx, 0)


def expiry_index(ts: pd.Series, i_entry: int, expiry_hours: float) -> int:
    """
    엔트리 index(i_entry)로부터 expiry_hours 후 시각보다 크거나 같은 첫 봉의 index를 반환.
    찾지 못하면 마지막 인덱스.
    """
    if i_entry >= len(ts) - 1:
        return len(ts) - 1
    t0 = ts.iloc[i_entry]
    target = to_utc_naive(t0) + pd.Timedelta(hours=expiry_hours)
    ts64 = ts.astype("datetime64[ns]")
    key = np.datetime64(target)
    idx = int(np.searchsorted(ts64, key, side="left"))
    if idx <= i_entry:
        idx = i_entry + 1
    if idx >= len(ts):
        idx = len(ts) - 1
    return idx


# -------------------- 시뮬레이션 --------------------
def simulate_symbol_breakout_close(
    sym: str,
    ohlcv: pd.DataFrame,
    sig_rows: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float,
    timeframe: str,
    expiry_hours: float
) -> pd.DataFrame:
    """
    - 엔트리: 신호봉 다음 봉의 '종가' (close)
    - TP/SL: intrabar 순서 불명 상황에서 보수적으로 SL 우선 체크 후 TP 체크
    - 만기: expiry_hours 내 TP/SL 미충족 시 만기 시점 '종가' 청산
    """
    if ohlcv is None or ohlcv.empty or len(sig_rows) == 0:
        return pd.DataFrame()

    ts = ohlcv["ts"]
    open_ = ohlcv["open"].to_numpy(dtype=float)
    high_ = ohlcv["high"].to_numpy(dtype=float)
    low__ = ohlcv["low"].to_numpy(dtype=float)
    close = ohlcv["close"].to_numpy(dtype=float)

    trades = []
    tf_min = timeframe_to_minutes(timeframe)

    for _, s in sig_rows.iterrows():
        sig_ts = to_utc_naive(s["ts"])
        i_sig = idx_of_bar(ts, sig_ts)
        i_entry = i_sig + 1
        if i_entry >= len(ohlcv):
            continue

        entry_px = float(close[i_entry])  # 신호봉 다음 봉의 종가로 진입
        if entry_px <= 0:
            continue
        tp_px = entry_px * (1.0 + tp_pct / 100.0)
        sl_px = entry_px * (1.0 - sl_pct / 100.0)

        i_exp = expiry_index(ts, i_entry, expiry_hours)

        exit_px = float(close[i_exp])     # 디폴트: 만기 종가
        exit_ts = ts.iloc[i_exp]
        reason = "expiry"

        # i_entry+1 부터 i_exp 까지 바-바 스캔
        for j in range(i_entry + 1, i_exp + 1):
            lo = float(low__[j])
            hi = float(high_[j])
            # 보수적으로 SL 우선
            if lo <= sl_px:
                exit_px = sl_px
                exit_ts = ts.iloc[j]
                reason = "SL"
                break
            if hi >= tp_px:
                exit_px = tp_px
                exit_ts = ts.iloc[j]
                reason = "TP"
                break

        gross = (exit_px / entry_px) - 1.0
        net = gross - fee_rt  # 라운드트립 수수료 차감

        trades.append({
            "symbol": sym,
            "signal_ts": sig_ts,
            "entry_ts": ts.iloc[i_entry],
            "exit_ts": exit_ts,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "fee_rt": fee_rt,
            "expiry_h": expiry_hours,
            "bar_tf_min": tf_min,
            "reason": reason,
            "gross": gross,
            "net": net,
        })

    return pd.DataFrame(trades)


def run_one_threshold(
    df_sig: pd.DataFrame,
    timeframe: str,
    expiry_h: float,
    tp: float,
    sl: float,
    fee_rt: float,
    touches_min: int,
    label: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    한 개의 최소 터치 기준(touches_min)으로 전체 시뮬 실행.
    label 은 결과 전략명 접미사.
    """
    # 1) 필터링: 롱 기준 → resistance 신호만 사용
    df = df_sig.copy()
    df = df[df["side"].str.lower() == "resistance"]

    # touches 숫자화
    df["touches"] = pd.to_numeric(df["touches"], errors="coerce").fillna(0).astype(int)
    df = df[df["touches"] >= touches_min]

    if df.empty:
        print(f"[BT] touches>={touches_min} 대상 신호가 없습니다.")
        return pd.DataFrame(), pd.DataFrame()

    syms = sorted(df["symbol"].dropna().unique().tolist())
    print(f"[BT] scenario: touches>={touches_min}, TP={tp} SL={sl}, expiry={expiry_h:.1f}h, symbols={len(syms)}")

    all_trades: List[pd.DataFrame] = []

    for sym in syms:
        rows = df[df["symbol"] == sym].sort_values("ts").reset_index(drop=True)
        ohlcv = get_ohlcv_safe(sym, timeframe)
        if ohlcv is None:
            continue
        tr = simulate_symbol_breakout_close(sym, ohlcv, rows, tp, sl, fee_rt, timeframe, expiry_h)
        if not tr.empty:
            tr["strategy"] = f"touch{touches_min}_{tp}/{sl}_{int(expiry_h)}h"
            all_trades.append(tr)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if trades.empty:
        return trades, pd.DataFrame()

    # 통계
    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "trades": len(g),
            "win_rate": float(np.mean(g["net"] > 0.0)) if len(g) else 0.0,
            "avg_net": float(np.mean(g["net"])) if len(g) else 0.0,
            "median_net": float(np.median(g["net"])) if len(g) else 0.0,
            "total_net": float(np.sum(g["net"])) if len(g) else 0.0,
        })

    stats = trades.groupby("strategy", as_index=False, dropna=False).apply(agg)
    return trades, stats


# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", type=str, help="TV signals CSV 경로 (columns: ts,event,side,level,touches,symbol,...)")
    ap.add_argument("--timeframe", type=str, default="15m")
    ap.add_argument("--expiry", type=str, default="24h")     # 예: 24h
    ap.add_argument("--tp", type=float, default=1.5)         # %
    ap.add_argument("--sl", type=float, default=1.0)         # %
    ap.add_argument("--fee", type=float, default=0.1/100.0)  # 라운드트립 수수료 (기본 0.1% = 0.001)
    ap.add_argument("--touches", type=int, default=2)
    ap.add_argument("--touches-alt", type=int, default=None)
    args = ap.parse_args()

    expiry_h = parse_expiry_hours(args.expiry)

    # 1) 시그널 로드 & UTC-naive 정규화
    if not os.path.exists(args.signals):
        raise SystemExit(f"signals 파일을 찾을 수 없습니다: {args.signals}")

    df_sig = pd.read_csv(args.signals)
    # 필수 컬럼 보정
    for c in ["ts","side","touches","symbol"]:
        if c not in df_sig.columns:
            raise SystemExit(f"signals CSV에 '{c}' 컬럼이 필요합니다.")

    df_sig["ts"] = series_to_utc_naive(df_sig["ts"])
    df_sig["symbol"] = df_sig["symbol"].astype(str).str.upper()

    syms = sorted(df_sig["symbol"].dropna().unique().tolist())
    print(f"[BT] symbols={len(syms)} signals(rows)={len(df_sig)} timeframe={args.timeframe}")
    print(f"[BT] scenario: touches>={args.touches}, TP={args.tp} SL={args.sl}, expiry={expiry_h:.1f}h")

    # 2) 기준1 실행
    trades1, stats1 = run_one_threshold(
        df_sig=df_sig,
        timeframe=args.timeframe,
        expiry_h=expiry_h,
        tp=args.tp,
        sl=args.sl,
        fee_rt=args.fee,
        touches_min=args.touches,
        label=f"touch{args.touches}"
    )

    # 3) 기준2(옵션) 실행
    trades2, stats2 = (pd.DataFrame(), pd.DataFrame())
    if args.touches_alt is not None:
        print(f"[BT] 추가 비교: touches>={args.touches_alt}")
        trades2, stats2 = run_one_threshold(
            df_sig=df_sig,
            timeframe=args.timeframe,
            expiry_h=expiry_h,
            tp=args.tp,
            sl=args.sl,
            fee_rt=args.fee,
            touches_min=args.touches_alt,
            label=f"touch{args.touches_alt}"
        )

    # 4) 저장/표시
    out_trades = pd.concat([x for x in [trades1, trades2] if not x.empty], ignore_index=True) if (not trades1.empty or not trades2.empty) else pd.DataFrame()
    out_stats  = pd.concat([x for x in [stats1, stats2] if not x.empty], ignore_index=True)  if (not stats1.empty or not stats2.empty)   else pd.DataFrame()

    if not out_trades.empty:
        trades_path = "./logs/bt_tv_breakout_trades.csv"
        out_trades.to_csv(trades_path, index=False, encoding="utf-8")
    else:
        trades_path = "(no trades)"

    if not out_stats.empty:
        stats_path = "./logs/bt_tv_breakout_stats.csv"
        out_stats.to_csv(stats_path, index=False, encoding="utf-8")
    else:
        stats_path = "(no stats)"

    print("\n=== TV Breakout (close-entry) Backtest ===")
    print(f"Trades saved: {trades_path}{'' if out_trades == '(no trades)' else f' (rows={len(out_trades)})'}")
    print(f"Stats  saved: {stats_path}{'' if out_stats  == '(no stats)'  else f' (rows={len(out_stats)})'}\n")

    if not out_stats.empty:
        cols = ["strategy","trades","win_rate","avg_net","median_net","total_net"]
        dfp = out_stats[cols].sort_values("avg_net", ascending=False)
        print(dfp.to_string(index=False))


if __name__ == "__main__":
    main()