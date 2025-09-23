# -*- coding: utf-8 -*-
"""
backtest_tv_breakout_close_sp.py

- TV signals 기반: 레벨 돌파 후 "신호봉 종가 기준 엔트리" (롱 온리)
- touches 임계 2/3 비교
- TP/SL % + 만료시간 (기본 24h)
- 수수료 왕복 0.1% (기본)

핵심 수정:
- 모든 시간 비교를 epoch nanoseconds(int64)로 통일 → tz-naive/aware 문제 제거
- 'Cannot pass tz with tzinfo...' 회피 (to_datetime(..., utc=True)만 사용)
- (no trades) 안전 처리
"""

import os
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import pyupbit


# -----------------------
# 시간 유틸
# -----------------------

def parse_expiry(exp: str) -> float:
    """
    '24h', '8h', '90m', '1d' → hours(float)
    """
    s = exp.strip().lower()
    if s.endswith("m"):
        return int(s[:-1]) / 60.0
    if s.endswith("h"):
        return float(s[:-1])
    if s.endswith("d"):
        return float(s[:-1]) * 24.0
    return float(s)


def series_ns(ts_series: pd.Series) -> np.ndarray:
    """
    ts(datetime64[ns, UTC] 또는 naive) -> epoch ns(int64) numpy array
    """
    if not pd.api.types.is_datetime64_any_dtype(ts_series):
        ts_series = pd.to_datetime(ts_series, utc=True)
    else:
        # tz-aware → UTC 유지, tz-naive면 UTC로 로컬라이즈
        if getattr(ts_series.dt, "tz", None) is None:
            ts_series = ts_series.dt.tz_localize("UTC")
        else:
            ts_series = ts_series.dt.tz_convert("UTC")
    # pandas 2.x: view('int64') OK
    return ts_series.view("int64").to_numpy()


def ts_to_utc(ts) -> pd.Timestamp:
    """
    단일 timestamp/문자열을 UTC-aware Timestamp로
    """
    t = pd.to_datetime(ts, utc=True)
    # 이미 utc=True를 사용했으니 tz-aware 보장
    return pd.Timestamp(t)


def idx_of_bar(ts_ns: np.ndarray, key_ts) -> int:
    """
    key_ts가 속하는 bar index (epoch ns 정렬배열에서 searchsorted)
    ts_ns: ascending epoch ns array
    """
    key_ns = int(pd.to_datetime(key_ts, utc=True).value)
    idx = int(np.searchsorted(ts_ns, key_ns, side="right")) - 1
    return max(0, idx)


# -----------------------
# 데이터 로드
# -----------------------

def get_ohlcv(symbol: str, tf: str = "15m", count: int = 500) -> pd.DataFrame | None:
    """
    Upbit OHLCV 로드 → ts(UTC aware), open/high/low/close/volume
    """
    try:
        df = pyupbit.get_ohlcv(symbol, interval=tf, count=count)
        if df is None or df.empty:
            print(f"[{symbol}] OHLCV empty")
            return None
        df = df.reset_index()
        # index 컬럼을 UTC aware로
        df["ts"] = pd.to_datetime(df["index"], utc=True)
        out = df[["ts", "open", "high", "low", "close", "volume"]].copy()
        return out
    except Exception as e:
        print(f"[{symbol}] OHLCV load error: {e}")
        return None


# -----------------------
# 시뮬레이션 (close-entry)
# -----------------------

def simulate_symbol_breakout_close(
    symbol: str,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float,
    timeframe: str,
    expiry_hours: float,
    touches_req: int = 2,
) -> pd.DataFrame:
    """
    - 엔트리: 신호가 발생한 bar의 '종가'로 진입
    - 슬/익: high/low 터치로 체결 가정
    - 만료: expiry_hours 후 첫 종가
    """
    trades = []
    # 시간축을 ns로
    ts_ns_arr = series_ns(ohlcv["ts"])

    for _, s in signals.iterrows():
        # 시그널 시각(UTC aware)
        sig_ts = ts_to_utc(s["ts"])

        # 터치 필터
        t = int(s.get("touches", 0) or 0)
        if t < touches_req:
            continue

        # 신호 bar index
        i_sig = idx_of_bar(ts_ns_arr, sig_ts)
        if i_sig < 0 or i_sig >= len(ohlcv) - 1:
            continue

        entry_px = float(ohlcv["close"].iloc[i_sig])

        # TP / SL
        tp_px = entry_px * (1 + tp_pct / 100.0)
        sl_px = entry_px * (1 - sl_pct / 100.0)

        # 만료 index (ns)
        expiry_ts = sig_ts + timedelta(hours=expiry_hours)
        exp_ns = int(expiry_ts.value)
        idx_exp = int(np.searchsorted(ts_ns_arr, exp_ns, side="left"))
        if idx_exp <= i_sig:
            continue

        df_future = ohlcv.iloc[i_sig + 1 : idx_exp + 1]
        if df_future.empty:
            continue

        exit_px, exit_reason = None, None
        for _, row in df_future.iterrows():
            lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])
            if lo <= sl_px:
                exit_px, exit_reason = sl_px, "SL"
                break
            if hi >= tp_px:
                exit_px, exit_reason = tp_px, "TP"
                break
        if exit_px is None:
            exit_px, exit_reason = float(df_future["close"].iloc[-1]), "EXP"

        net = (exit_px - entry_px) / entry_px
        net -= fee_rt

        trades.append(
            {
                "symbol": symbol,
                "sig_ts": sig_ts.isoformat(),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "exit_reason": exit_reason,
                "tp": tp_pct,
                "sl": sl_pct,
                "expiry_h": expiry_hours,
                "touches_req": touches_req,
                "touches": t,
                "net": net,
            }
        )

    return pd.DataFrame(trades)


# -----------------------
# 집계/출력
# -----------------------

def agg_stats(g: pd.DataFrame) -> pd.Series:
    if g is None or g.empty:
        return pd.Series(
            {
                "trades": 0,
                "win_rate": np.nan,
                "avg_net": np.nan,
                "median_net": np.nan,
                "total_net": np.nan,
            }
        )
    return pd.Series(
        {
            "trades": len(g),
            "win_rate": float(np.mean(g["net"] > 0)) if len(g) else np.nan,
            "avg_net": float(g["net"].mean()) if len(g) else np.nan,
            "median_net": float(g["net"].median()) if len(g) else np.nan,
            "total_net": float(g["net"].sum()) if len(g) else np.nan,
        }
    )


def summarize(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    if trades is None or trades.empty:
        print(f"(no trades for {label})")
        return pd.DataFrame(columns=["strategy", "trades", "win_rate", "avg_net", "median_net", "total_net"])
    stats = trades.groupby("strategy", as_index=False, dropna=False).apply(agg_stats)
    print(f"\n=== {label} ===")
    print(stats.to_string(index=False))
    return stats


# -----------------------
# 메인
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="24h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)  # 왕복 0.1%
    ap.add_argument("--touches", type=int, default=2)
    ap.add_argument("--touches-alt", type=int, default=3)
    args = ap.parse_args()

    expiry_hours = parse_expiry(args.expiry)

    # 시그널 로드
    df_sig = pd.read_csv(args.signals)
    if "ts" not in df_sig.columns or "symbol" not in df_sig.columns:
        raise ValueError("signals CSV must contain at least 'ts' and 'symbol' columns.")
    # tz-aware로 통일
    df_sig["ts"] = pd.to_datetime(df_sig["ts"], utc=True)

    # 심볼 세트
    symbols = sorted(df_sig["symbol"].dropna().unique())
    print(f"[BT] symbols={len(symbols)} signals(rows)={len(df_sig)} timeframe={args.timeframe}")

    # --- touches 기준 1 (기본) ---
    trades_all = []
    print(f"[BT] scenario: touches>={args.touches}, TP={args.tp} SL={args.sl}, expiry={expiry_hours}h")
    for sym in symbols:
        ohlcv = get_ohlcv(sym, args.timeframe, count=500)
        if ohlcv is None:
            continue
        rows = df_sig.loc[df_sig["symbol"] == sym]
        tr = simulate_symbol_breakout_close(
            sym,
            ohlcv,
            rows,
            tp_pct=args.tp,
            sl_pct=args.sl,
            fee_rt=args.fee,
            timeframe=args.timeframe,
            expiry_hours=expiry_hours,
            touches_req=args.touches,
        )
        if not tr.empty:
            tr["strategy"] = f"touch{args.touches}_{args.tp}/{args.sl}_{args.expiry}"
            trades_all.append(tr)

    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    stats = summarize(trades, f"touches>={args.touches}")

    # --- touches 기준 2 (대안: touches_alt) ---
    print("[BT] 추가 비교: touches>={}".format(args.touches_alt))
    trades_all_alt = []
    for sym in symbols:
        ohlcv = get_ohlcv(sym, args.timeframe, count=500)
        if ohlcv is None:
            continue
        rows = df_sig.loc[df_sig["symbol"] == sym]
        tr = simulate_symbol_breakout_close(
            sym,
            ohlcv,
            rows,
            tp_pct=args.tp,
            sl_pct=args.sl,
            fee_rt=args.fee,
            timeframe=args.timeframe,
            expiry_hours=expiry_hours,
            touches_req=args.touches_alt,
        )
        if not tr.empty:
            tr["strategy"] = f"touch{args.touches_alt}_{args.tp}/{args.sl}_{args.expiry}"
            trades_all_alt.append(tr)

    trades_alt = pd.concat(trades_all_alt, ignore_index=True) if trades_all_alt else pd.DataFrame()
    stats_alt = summarize(trades_alt, f"touches>={args.touches_alt}")

    # 저장
    os.makedirs("logs", exist_ok=True)
    trades_path = "./logs/bt_tv_breakout_trades.csv"
    stats_path = "./logs/bt_tv_breakout_stats.csv"
    trades_alt_path = "./logs/bt_tv_breakout_trades_alt.csv"
    stats_alt_path = "./logs/bt_tv_breakout_stats_alt.csv"

    if trades is not None and not trades.empty:
        trades.to_csv(trades_path, index=False)
    else:
        # 비어 있으면 헤더만
        pd.DataFrame(columns=["symbol","sig_ts","entry_px","exit_px","exit_reason","tp","sl","expiry_h","touches_req","touches","net","strategy"]).to_csv(trades_path, index=False)

    if stats is not None and not stats.empty:
        stats.to_csv(stats_path, index=False)
    else:
        pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"]).to_csv(stats_path, index=False)

    if trades_alt is not None and not trades_alt.empty:
        trades_alt.to_csv(trades_alt_path, index=False)
    else:
        pd.DataFrame(columns=["symbol","sig_ts","entry_px","exit_px","exit_reason","tp","sl","expiry_h","touches_req","touches","net","strategy"]).to_csv(trades_alt_path, index=False)

    if stats_alt is not None and not stats_alt.empty:
        stats_alt.to_csv(stats_alt_path, index=False)
    else:
        pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"]).to_csv(stats_alt_path, index=False)

    print("\n[BT] 저장 완료.")
    print(f"  → {trades_path}")
    print(f"  → {stats_path}")
    print(f"  → {trades_alt_path}")
    print(f"  → {stats_alt_path}")


if __name__ == "__main__":
    main()