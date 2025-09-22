# backtest_tv_multi_strategies.py
# --------------------------------
# TradingView(폴 지표) 알람만을 대상으로 한 멀티-전략 백테스터
# - Long only
# - Entry: 다음 봉 "시가"
# - Exit: TP/SL 터치 또는 24h 만기(종가 청산)
# - Fee: 0.05% 진입 + 0.05% 청산 = 왕복 0.1%
# - 시그널 기준: TV 알람 중 'support' 사이드만 진입 대상으로 사용
#
# 실행 예시:
#   python backtest_tv_multi_strategies.py ./logs/signals_tv.csv --timeframe 15m
#
# 결과:
#   ./logs/bt_tv_trades.csv   (개별 체결 내역)
#   ./logs/bt_tv_stats.csv    (전략별 요약 통계)



import os
import argparse
from functools import lru_cache
from math import ceil

import numpy as np
import pandas as pd

# upbit OHLCV 로더는 프로젝트의 sr_engine.data 사용
from sr_engine.data import get_ohlcv


# ------------------------ 유틸 ------------------------
def ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def ensure_ts_column(df: pd.DataFrame) -> pd.DataFrame:
    """ts 컬럼을 UTC-aware pandas.Timestamp로 보정."""
    df = df.copy()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            df["ts"] = idx
        else:
            # 마지막 보루: 문자열/숫자 → 날짜
            for c in ("timestamp", "time", "datetime", "date", "ts"):
                if c in df.columns:
                    df["ts"] = pd.to_datetime(df[c], utc=True, errors="coerce")
                    break
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def tf_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 15

@lru_cache(maxsize=None)
def get_ohlcv_cached(symbol: str, timeframe: str) -> pd.DataFrame:
    df = get_ohlcv(symbol, timeframe)
    df = ensure_ts_column(df)
    # 컬럼 통일 (pyupbit는 open/high/low/close/volume/value 기준)
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise RuntimeError(f"OHLCV에 '{c}'가 없습니다: {symbol} {timeframe}")
    return df

def next_bar_open_idx(ohlcv: pd.DataFrame, signal_ts: pd.Timestamp) -> int | None:
    """시그널 직후 '다음 봉 시가' 인덱스(= 시뮬 진입 인덱스) 반환."""
    ts = ohlcv["ts"].to_numpy(dtype="datetime64[ns]")
    sig = np.datetime64(pd.Timestamp(signal_ts, tz="UTC").to_datetime64())
    idx = ts.searchsorted(sig, side="right")  # signal_ts 이후 첫 봉
    if idx >= len(ts):
        return None
    return int(idx)

def bars_for_24h(timeframe: str) -> int:
    return max(1, ceil(24 * 60 / tf_to_minutes(timeframe)))


# ------------------------ 시그널 로드/필터 ------------------------
def load_tv_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 표준 컬럼 기대: ts,event,side,level,touches,symbol,timeframe,extra,source,host,message
    # 없을 수도 있으니 방어적으로 처리
    needed = {"ts", "event", "side", "symbol"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"필수 컬럼 누락: {missing}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "symbol", "side"]).reset_index(drop=True)

    # TV만 사용: source가 TV이거나, message/host에 TradingView 흔적이 있는 것 우선
    # 단, 이미 split한 파일이라면 전체가 TV일 수 있음 → 일단 모두 사용
    # 롱 전용: support 신호만 사용
    df = df[df["side"].str.lower() == "support"].copy()

    # 이벤트 필터(폴 지표): level2_detected, price_in_box 정도를 주로 사용
    df["event"] = df["event"].str.lower()
    tv_events = {"level2_detected", "price_in_box"}  # 필요시 확장
    df = df[df["event"].isin(tv_events)].copy()

    # 심볼/시간 프레임 정리
    if "timeframe" not in df.columns or df["timeframe"].isna().all():
        df["timeframe"] = "15m"  # 기본값 (인자가 있으면 그걸로 대체 가능)
    else:
        # 혼재되어 있어도 이 스크립트에서는 공통 timeframe 인자를 우선 사용
        pass

    # 컬럼 최소화
    return df[["ts", "event", "side", "symbol", "timeframe"]].sort_values("ts").reset_index(drop=True)


# ------------------------ 시뮬레이션 ------------------------
def simulate_symbol(
    sym: str,
    df_sig: pd.DataFrame,
    timeframe: str,
    tp: float,
    sl: float,
    fee_one_side: float = 0.0005,
) -> list[dict]:
    """
    sym에 대한 모든 TV 시그널을 순서대로 시뮬.
      - entry: 다음 봉 시가
      - exit: TP/SL 터치 or 24h 만기(마지막 봉 종가)
      - net = gross - (fee_in + fee_out)
    반환: trade dict 리스트
    """
    ohlcv = get_ohlcv_cached(sym, timeframe)
    trades = []
    max_bars = bars_for_24h(timeframe)

    ts_arr = ohlcv["ts"].to_numpy()
    open_arr = ohlcv["open"].to_numpy(dtype=float)
    high_arr = ohlcv["high"].to_numpy(dtype=float)
    low_arr  = ohlcv["low"].to_numpy(dtype=float)
    close_arr = ohlcv["close"].to_numpy(dtype=float)

    for _, s in df_sig.iterrows():
        sig_ts = pd.Timestamp(s["ts"], tz="UTC")
        i = next_bar_open_idx(ohlcv, sig_ts)
        if i is None or i >= len(ohlcv):
            continue

        entry_ts = pd.Timestamp(ts_arr[i]).tz_localize("UTC") if pd.Timestamp(ts_arr[i]).tzinfo is None else pd.Timestamp(ts_arr[i]).tz_convert("UTC")
        entry_px = float(open_arr[i])
        target_px = entry_px * (1.0 + tp)
        stop_px   = entry_px * (1.0 - sl)

        exit_idx = None
        exit_px = None
        exit_reason = None

        # i번째 봉부터 최대 max_bars 동안 진행 (i 포함)
        # intrabar 동시 터치시 보수적 가정: SL 우선 체결(sl_wins_on_both)
        last_idx = min(len(ohlcv) - 1, i + max_bars - 1)  # 만기 봉 인덱스
        for j in range(i, last_idx + 1):
            hi = float(high_arr[j]); lo = float(low_arr[j])

            hit_tp = hi >= target_px
            hit_sl = lo <= stop_px

            if hit_tp and hit_sl:
                # 보수적으로 SL 우선 (보다 엄격)
                exit_idx = j
                exit_px = stop_px
                exit_reason = "SL_both_hit"
                break
            elif hit_sl:
                exit_idx = j
                exit_px = stop_px
                exit_reason = "SL"
                break
            elif hit_tp:
                exit_idx = j
                exit_px = target_px
                exit_reason = "TP"
                break

        if exit_idx is None:
            # 만기: 마지막 봉 종가
            exit_idx = last_idx
            exit_px = float(close_arr[exit_idx])
            exit_reason = "EXPIRY"

        gross = (exit_px / entry_px) - 1.0
        net = gross - (fee_one_side + fee_one_side)  # 왕복

        trades.append({
            "symbol": sym,
            "signal_ts": sig_ts.isoformat(),
            "entry_ts": pd.Timestamp(ts_arr[i]).isoformat(),
            "entry_px": entry_px,
            "exit_ts": pd.Timestamp(ts_arr[exit_idx]).isoformat(),
            "exit_px": exit_px,
            "exit_reason": exit_reason,
            "tp": tp,
            "sl": sl,
            "ret_gross": gross,
            "ret_net": net,
        })

    return trades


def simulate_all(df_sig: pd.DataFrame, timeframe: str, strategies: list[tuple[str, float, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    여러 전략을 순차로 돌려 트레이드/통계를 반환.
    strategies: [(name, tp, sl), ...]
    """
    out_trades = []

    symbols = sorted(df_sig["symbol"].unique().tolist())
    for name, tp, sl in strategies:
        for sym in symbols:
            rows = df_sig[df_sig["symbol"] == sym]
            tr = simulate_symbol(sym, rows, timeframe, tp, sl)
            for t in tr:
                t["strategy"] = name
            out_trades.extend(tr)
        print(f"[BT] {name} done. sym={len(symbols)} trades={sum(1 for t in out_trades if t['strategy']==name)}")

    trades_df = pd.DataFrame(out_trades)
    if trades_df.empty:
        stats_df = pd.DataFrame(columns=["strategy", "trades", "win_rate", "avg_net", "total_net", "median_net"])
        return trades_df, stats_df

    # 통계
    def _agg(g):
        trades = len(g)
        wins = int((g["ret_net"] > 0).sum())
        win_rate = wins / trades if trades else 0.0
        return pd.Series({
            "trades": trades,
            "win_rate": round(win_rate, 4),
            "avg_net": round(float(g["ret_net"].mean()), 6),
            "median_net": round(float(g["ret_net"].median()), 6),
            "total_net": round(float(g["ret_net"].sum()), 6),
        })

    stats_df = trades_df.groupby("strategy", as_index=False).apply(_agg, include_groups=False)
    stats_df = stats_df.sort_values(["total_net", "win_rate"], ascending=[False, False])
    return trades_df, stats_df


# ------------------------ 메인 ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", nargs="?", default="./logs/signals_tv.csv", help="TV 전용 시그널 CSV 경로")
    ap.add_argument("--timeframe", default="15m", help="OHLCV 타임프레임 (기본 15m)")
    ap.add_argument("--out-trades", default="./logs/bt_tv_trades.csv", help="체결 결과 CSV")
    ap.add_argument("--out-stats", default="./logs/bt_tv_stats.csv", help="전략 통계 CSV")
    args = ap.parse_args()

    ensure_dir(args.out_trades)
    ensure_dir(args.out_stats)

    df_sig = load_tv_signals(args.signals)
    # 공통 timeframe 강제 (혼재되어 있더라도 인자로 통일)
    df_sig["timeframe"] = args.timeframe

    # ===== 전략 세트 (6개) =====
    strategies = [
        ("stable_1.5/1.0",   0.015,  0.010),
        ("aggressive_2.0/1.25", 0.020,  0.0125),
        ("scalp_1.0/0.75",  0.010,  0.0075),
        ("mid_1.25/1.0",    0.0125, 0.010),
        ("mid_1.75/1.25",   0.0175, 0.0125),
        ("tight_0.8/0.8",   0.008,  0.008),
    ]

    trades_df, stats_df = simulate_all(df_sig, args.timeframe, strategies)

    trades_df.to_csv(args.out_trades, index=False)
    stats_df.to_csv(args.out_stats, index=False)

    print("\n=== TV Backtest (Long-only / 24h expiry / 0.1% fees) ===")
    if trades_df.empty:
        print("No trades produced.")
        return

    print(f"Trades saved: {args.out_trades}  (rows={len(trades_df)})")
    print(f"Stats  saved: {args.out_stats}  (rows={len(stats_df)})\n")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
