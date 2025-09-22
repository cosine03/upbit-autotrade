# -*- coding: utf-8 -*-
"""
backtest_tv_entry_close.py
- TV(폴) 신호 대상 백테스트
- 엔트리: 알람 발생 시점의 '장중 가격(1분 해상도 근사)' 이하일 때만 진입 (롱)
- 만기: 4시간/8시간 (둘 다 수행)
- 수수료: 왕복 0.1% (0.05% + 0.05%)
- 데이터: sr_engine.data.get_ohlcv 사용 (15m 본체 + 1m 근사용)

실행:
  python backtest_tv_entry_close.py ./logs/signals_tv.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from sr_engine.data import get_ohlcv  # 15m/1m OHLCV 로더

FEES_RT = 0.001  # 왕복 0.1%

# -------- 공통 유틸 --------
def to_utc_ts(x) -> pd.Timestamp:
    """모든 타임스탬프를 tz-aware(UTC)로 통일"""
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.Timestamp(x, tz="UTC")

def next_bar_open_idx(ohlcv_15m: pd.DataFrame, signal_ts: pd.Timestamp) -> int:
    """시그널 직후 '다음 봉' 오픈 인덱스(15m 데이터 기준)"""
    ts = pd.to_datetime(ohlcv_15m["ts"], utc=True).to_numpy()
    sig_ts = to_utc_ts(signal_ts)
    return int(np.searchsorted(ts, sig_ts.to_datetime64(), side="right"))

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["ts"] = df.index
        else:
            raise RuntimeError("OHLCV에 ts가 없습니다.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)

# -------- 신호가(장중가) 근사: 1분 해상도 --------
def get_signal_price(symbol: str, signal_ts: pd.Timestamp) -> float:
    """
    알람 발생 시각의 장중 가격을 1분 OHLCV로 근사.
    - signal_ts 시각과 같거나 직전의 1분 캔들 close를 '신호가'로 사용
    - 1분 데이터가 없거나 못 찾으면 백업으로 '신호가 포함 15m 캔들 close' 사용
    """
    sig = to_utc_ts(signal_ts)
    # 1분 캔들 넉넉히 가져오기 (최근 400분)
    df1 = get_ohlcv(symbol, "1m")  # sr_engine이 내부에서 적당량 반환
    df1 = ensure_ts(df1)

    # 신호 시각 이하인 마지막 1분 봉
    m = df1["ts"] <= sig
    if m.any():
        close_price = float(df1.loc[m, "close"].iloc[-1])
        return close_price

    # 백업: 15m close
    df15 = get_ohlcv(symbol, "15m")
    df15 = ensure_ts(df15)
    # signal_ts가 포함된(또는 직전) 15m 봉 close
    m15 = df15["ts"] <= sig
    if m15.any():
        return float(df15.loc[m15, "close"].iloc[-1])
    # 마지막 백업: 최신 close
    return float(df15["close"].iloc[-1])

# -------- 시뮬(한 심볼) --------
def simulate_symbol(
    symbol: str,
    rows: pd.DataFrame,
    timeframe: str,
    tp: float,
    sl: float,
    expiry_hours: int,
) -> pd.DataFrame:
    """
    rows: 해당 symbol의 시그널들(ts, side 등 포함)
    엔트리 로직:
      - 알람 시각의 '장중 신호가' = signal_price 산출(1m)
      - 다음 15m 봉부터 순차 검사:
          * 만약 해당 봉 open <= signal_price → open 체결
          * 아니고, 봉 low <= signal_price → signal_price 체결
          * 아니면 스킵(진입 안함)
      - 진입 후 TP/SL/만기(종가청산) 처리
    """
    ohlcv = get_ohlcv(symbol, timeframe)
    ohlcv = ensure_ts(ohlcv)

    trades = []
    for _, s in rows.iterrows():
        sig_ts = to_utc_ts(s["ts"])

        # 장중 신호가(1m 근사)
        signal_price = get_signal_price(symbol, sig_ts)

        # 다음 15m 봉 open부터 시작
        i = next_bar_open_idx(ohlcv, sig_ts)
        if i >= len(ohlcv):
            continue

        # 진입 시점 찾기: price <= signal_price 되는 첫 봉
        entry_idx = None
        entry_price = None

        j = i
        while j < len(ohlcv):
            op = float(ohlcv["open"].iloc[j])
            lo = float(ohlcv["low"].iloc[j])
            hi = float(ohlcv["high"].iloc[j])  # TP/SL 판단에 쓸 수도 있어 미리 가져옴

            if op <= signal_price:
                entry_price = op  # 시가가 이미 신호가 이하 → 시가 체결
                entry_idx = j
                break
            if lo <= signal_price:
                entry_price = signal_price  # 봉 중간에 내려와 신호가 히트 → 신호가 체결
                entry_idx = j
                break

            # 이번 봉에서 조건 미충족 → 다음 봉로 이동
            j += 1

        if entry_idx is None:
            # 만기까지도 신호가 이하로 안 왔다면 미진입
            continue

        # 만기 시각
        exit_deadline = sig_ts + pd.Timedelta(hours=expiry_hours)

        # 진입 이후 TP/SL/만기
        exit_price = None
        k = entry_idx
        while k < len(ohlcv) and ohlcv["ts"].iloc[k] <= exit_deadline:
            hi = float(ohlcv["high"].iloc[k])
            lo = float(ohlcv["low"].iloc[k])

            tp_price = entry_price * (1 + tp / 100.0)
            sl_price = entry_price * (1 - sl / 100.0)

            if hi >= tp_price:
                exit_price = tp_price
                break
            if lo <= sl_price:
                exit_price = sl_price
                break
            k += 1

        # 만기 청산(종가)
        if exit_price is None:
            # 만기 직후 첫 캔들 또는 만기 포함 봉의 종가로 처리
            if k < len(ohlcv):
                exit_price = float(ohlcv["close"].iloc[k])
            else:
                exit_price = float(ohlcv["close"].iloc[-1])

        net = (exit_price / entry_price - 1.0) - FEES_RT

        trades.append({
            "symbol": symbol,
            "signal_ts": sig_ts.isoformat(),
            "entry_ts": ohlcv["ts"].iloc[entry_idx].isoformat(),
            "exit_ts": ohlcv["ts"].iloc[min(k, len(ohlcv)-1)].isoformat(),
            "signal_price": signal_price,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "tp": tp,
            "sl": sl,
            "expiry_h": expiry_hours,
            "net": net,
        })

    return pd.DataFrame(trades)

# -------- 시뮬(전 체) --------
def simulate_all(df_sig: pd.DataFrame, timeframe: str, strategies: List[Tuple[str, float, float, int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    strategies: [(name, tp%, sl%, expiry_hours), ...]
    """
    out_trades = []
    for name, tp, sl, expiry_h in strategies:
        trades_all = []
        for sym, rows in df_sig.groupby("symbol"):
            tr = simulate_symbol(sym, rows, timeframe, tp, sl, expiry_h)
            trades_all.append(tr)
        trades_df = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
        trades_df["strategy"] = name
        print(f"[BT] {name} done. symbols={df_sig['symbol'].nunique()} trades={len(trades_df)}")
        out_trades.append(trades_df)

    trades = pd.concat(out_trades, ignore_index=True) if out_trades else pd.DataFrame()

    def agg_stats(g: pd.DataFrame) -> pd.Series:
        wins = (g["net"] > 0).sum()
        return pd.Series({
            "trades": len(g),
            "win_rate": wins / max(1, len(g)),
            "avg_net": g["net"].mean() if len(g) else 0.0,
            "median_net": g["net"].median() if len(g) else 0.0,
            "total_net": g["net"].sum() if len(g) else 0.0,
        })

    stats = trades.groupby("strategy", as_index=False, dropna=False).apply(agg_stats)
    return trades, stats

# -------- 메인 --------
def load_tv_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 표준화
    must = ["ts","event","side","symbol"]
    for c in must:
        if c not in df.columns:
            raise RuntimeError(f"signals_tv.csv에 '{c}' 컬럼이 없습니다.")
    # TV 알람만 사용 (side 무관), ts 표준화
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    # 필요시 필터링(예: price_in_box, level2_detected 등) — 지금은 전체 사용
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest_tv_entry_close.py ./logs/signals_tv.csv")
        sys.exit(1)

    path = sys.argv[1]
    df_sig = load_tv_signals(path)

    # 전략: 4h/8h 만기 둘 다
    STRATS = [
        ("stable_1.5/1.0_4h", 1.5, 1.0, 4),
        ("aggressive_2.0/1.25_4h", 2.0, 1.25, 4),
        ("scalp_1.0/0.75_4h", 1.0, 0.75, 4),
        ("mid_1.25/1.0_4h", 1.25, 1.0, 4),
        ("mid_1.75/1.25_4h", 1.75, 1.25, 4),
        ("tight_0.8/0.8_4h", 0.8, 0.8, 4),

        ("stable_1.5/1.0_8h", 1.5, 1.0, 8),
        ("aggressive_2.0/1.25_8h", 2.0, 1.25, 8),
        ("scalp_1.0/0.75_8h", 1.0, 0.75, 8),
        ("mid_1.25/1.0_8h", 1.25, 1.0, 8),
        ("mid_1.75/1.25_8h", 1.75, 1.25, 8),
        ("tight_0.8/0.8_8h", 0.8, 0.8, 8),
    ]

    trades_df, stats_df = simulate_all(df_sig, "15m", STRATS)

    os.makedirs("./logs", exist_ok=True)
    trades_path = "./logs/bt_tv_entryclose_trades.csv"
    stats_path  = "./logs/bt_tv_entryclose_stats.csv"
    trades_df.to_csv(trades_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print("\n=== TV Backtest (Entry = price <= signal_price(1m) / Long-only / fees=0.1% RT) ===")
    print(f"Trades saved: {trades_path} (rows={len(trades_df)})")
    print(f"Stats  saved: {stats_path} (rows={len(stats_df)})\n")
    print(stats_df.sort_values(["strategy"]).to_string(index=False))

if __name__ == "__main__":
    main()
