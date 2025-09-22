# -*- coding: utf-8 -*-
"""
backtest_tv_entry_close_mp.py
- 대상: TradingView(폴) 알람(logs/signals_tv.csv 권장)
- 진입 규칙 A(커스텀): 알람 발생 시점의 즉시가격(= 시그널이 발생한 봉의 종가) 이하로
  이후 가격이 내려오는 첫 번째 시점(다음 봉들)에서 진입.
  구현: 시그널 직후부터 스캔하며, '저가 <= alert_price'인 첫 봉에서
       entry = max(open, alert_price) 로 체결(보수적 체결 가정).
- 만기: 4시간 / 8시간(둘 다 수행)
- 포지션: Long only
- 수수료: 왕복 0.1% (0.001)
- 멀티프로세싱으로 심볼 병렬 처리
- 기본적으로 KRW-BTC, KRW-ETH는 제외 (옵션으로 변경 가능)
"""

import os
import argparse
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# OHLCV 로더 (pyupbit 래핑되어 있다고 가정: sr_engine.data.get_ohlcv)
from sr_engine.data import get_ohlcv

load_dotenv()

# -----------------------------
# 설정
# -----------------------------
FEE_RT = 0.001  # 왕복 0.1%
DEFAULT_TIMEFRAME = "15m"
DEFAULT_EXCLUDE = {"KRW-BTC", "KRW-ETH"}

STRATS: List[Tuple[float, float]] = [
    (1.5, 1.0),   # stable
    (2.0, 1.25),  # aggressive
    (1.0, 0.75),  # scalp
    (1.25, 1.0),  # mid
    (1.75, 1.25), # mid2
    (0.8, 0.8),   # tight
]

EXPIRIES_H = [4, 8]

CACHE_DIR = "./cache_bt_tv"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# 유틸
# -----------------------------
def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        else:
            for k in ("timestamp","time","datetime","date"):
                if k in df.columns:
                    ts = pd.to_datetime(df[k], errors="coerce", utc=True)
                    df["ts"] = ts
                    break
        if "ts" not in df.columns:
            raise RuntimeError("OHLCV에 ts/timestamp가 없습니다.")
    ts = pd.to_datetime(df["ts"], utc=True)
    df["ts"] = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    return df.sort_values("ts").reset_index(drop=True)

def tf_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 15

def get_ohlcv_cached(symbol: str, timeframe: str) -> pd.DataFrame:
    fn = os.path.join(CACHE_DIR, f"ohlcv_{symbol.replace('-','_')}_{timeframe}.csv")
    if os.path.exists(fn):
        df = pd.read_csv(fn)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return ensure_ts(df)
    df = get_ohlcv(symbol, timeframe)
    df = ensure_ts(df)
    # 필요한 최소 컬럼만 저장
    df[["ts","open","high","low","close","volume"]].to_csv(fn, index=False)
    return df

def next_bar_index(ts_series: pd.Series, signal_ts: pd.Timestamp) -> int:
    # ts_series: UTC tz-aware, signal_ts: tz-aware(UTC)
    ts_np = ts_series.to_numpy(dtype="datetime64[ns]")
    return int(np.searchsorted(ts_np, signal_ts.to_datetime64(), side="right"))

def bars_until(ts_series: pd.Series, start_idx: int, hours: int, tf_min: int) -> int:
    if start_idx >= len(ts_series):
        return start_idx
    start_ts = ts_series.iloc[start_idx]
    end_ts = start_ts + pd.Timedelta(hours=hours)
    ts_np = ts_series.to_numpy(dtype="datetime64[ns]")
    return int(np.searchsorted(ts_np, end_ts.to_datetime64(), side="right"))

# -----------------------------
# 시뮬레이션 핵심
# -----------------------------
def simulate_symbol(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    args = (symbol, df_sig_sym, timeframe, strategies, expiries_h)
    반환: (trades_df, stats_df)
    """
    symbol, df_sig_sym, timeframe, strategies, expiries_h = args

    ohlcv = get_ohlcv_cached(symbol, timeframe)
    ts = ohlcv["ts"]
    tfmin = tf_minutes(timeframe)

    trades_rows = []

    # 알람 발생 봉의 종가를 alert_price로 사용
    # 진입: 시그널 직후부터 스캔하며 low <= alert_price인 첫 봉에서
    #       entry = max(open, alert_price)
    for _, s in df_sig_sym.iterrows():
        sig_ts = pd.to_datetime(s["ts"], utc=True)
        # 시그널이 속한 봉 index 계산(다음 봉 오픈부터 스캔)
        i0 = next_bar_index(ts, sig_ts)
        if i0 >= len(ohlcv):
            continue

        # 알람 발생 시점이 속한 "직전 봉"의 종가를 alert_price로
        prev_idx = i0 - 1
        if prev_idx < 0:
            continue
        alert_price = float(ohlcv["close"].iloc[prev_idx])

        for tp, sl in strategies:
            for expiry_h in expiries_h:
                # 엔트리 탐색
                entry_idx = None
                entry_price = None
                end_idx = bars_until(ts, i0, expiry_h, tfmin)

                for i in range(i0, min(end_idx, len(ohlcv))):
                    lo = float(ohlcv["low"].iloc[i])
                    op = float(ohlcv["open"].iloc[i])
                    if lo <= alert_price:
                        entry_idx = i
                        entry_price = max(op, alert_price)
                        break

                if entry_idx is None:
                    # 미체결
                    trades_rows.append({
                        "symbol": symbol,
                        "ts_signal": sig_ts.isoformat(),
                        "ts_entry": None,
                        "ts_exit": ts.iloc[min(end_idx, len(ohlcv)-1)].isoformat(),
                        "entry": np.nan,
                        "exit": np.nan,
                        "gross": 0.0,
                        "net": -FEE_RT,     # 주문/취소 수수료 고려하지 않음, net= -fee 라고 간주하지 않고 0으로 둘 수도 있음
                        "tp": tp, "sl": sl, "expiry_h": expiry_h,
                        "status": "no_fill",
                    })
                    continue

                # TP/SL/만기 청산
                target = entry_price * (1 + tp/100.0)
                stop   = entry_price * (1 - sl/100.0)

                exit_idx = None
                exit_price = None
                status = "expiry"

                for i in range(entry_idx, min(end_idx, len(ohlcv))):
                    hi = float(ohlcv["high"].iloc[i])
                    lo = float(ohlcv["low"].iloc[i])

                    hit_tp = hi >= target
                    hit_sl = lo <= stop

                    if hit_tp and hit_sl:
                        # 같은 봉에서 둘 다 맞으면 보수적으로 SL 우선 처리 가능
                        # 여기서는 TP 우선 가정(원하면 반대로 바꿔도 됨)
                        exit_idx = i
                        exit_price = target
                        status = "tp"
                        break
                    elif hit_tp:
                        exit_idx = i
                        exit_price = target
                        status = "tp"
                        break
                    elif hit_sl:
                        exit_idx = i
                        exit_price = stop
                        status = "sl"
                        break

                if exit_idx is None:
                    # 만기 종가
                    exit_idx = min(end_idx-1, len(ohlcv)-1)
                    exit_price = float(ohlcv["close"].iloc[exit_idx])
                    status = "expiry"

                gross = (exit_price - entry_price) / entry_price
                net = gross - FEE_RT

                trades_rows.append({
                    "symbol": symbol,
                    "ts_signal": sig_ts.isoformat(),
                    "ts_entry": ts.iloc[entry_idx].isoformat(),
                    "ts_exit": ts.iloc[exit_idx].isoformat(),
                    "entry": entry_price,
                    "exit": exit_price,
                    "gross": gross,
                    "net": net,
                    "tp": tp, "sl": sl, "expiry_h": expiry_h,
                    "status": status,
                })

    trades_df = pd.DataFrame(trades_rows)
    if trades_df.empty:
        return trades_df, pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"])

    trades_df["strategy"] = trades_df.apply(lambda r: f"{r['tp']:.2g}/{r['sl']:.2g}_{int(r['expiry_h'])}h", axis=1)
    stats = (
        trades_df
        .groupby("strategy")
        .agg(
            trades=("net","count"),
            win_rate=("net", lambda s: float((s>0).mean())),
            avg_net=("net","mean"),
            median_net=("net","median"),
            total_net=("net","sum"),
        )
        .reset_index()
        .sort_values("total_net", ascending=False)
    )
    return trades_df, stats

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="TV alerts backtest (MP, entry on dip to alert price).")
    ap.add_argument("signals", nargs="?", default="./logs/signals_tv.csv", help="TV signals CSV path")
    ap.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    ap.add_argument("--exclude", default="KRW-BTC,KRW-ETH", help="심볼 제외(콤마 구분)")
    ap.add_argument("--processes", type=int, default=max(1, cpu_count()-1))
    ap.add_argument("--outdir", default="./logs", help="결과 저장 폴더")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.signals)
    # 필요한 컬럼 정규화
    need_cols = ["ts","symbol"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"signals에 {c} 컬럼이 없습니다.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["symbol"] = df["symbol"].astype(str)

    # BTC/ETH 제외 (기본)
    excl = {s.strip().upper() for s in args.exclude.split(",") if s.strip()}
    df = df[~df["symbol"].str.upper().isin(excl)].copy()

    # 알람 의미있는 이벤트만(선택): price_in_box/level_detected 등 제한하고 싶으면 필터 추가 가능
    # 여기서는 TV 파일 그대로 사용

    # 심볼별 그룹핑
    tasks = []
    for sym, rows in df.groupby("symbol"):
        rows = rows.sort_values("ts").reset_index(drop=True)
        tasks.append((sym, rows, args.timeframe, STRATS, EXPIRIES_H))

    if not tasks:
        print("No symbols to run (after exclusion).")
        return

    print(f"[BT-MP] start. symbols={len(tasks)} exclude={sorted(excl)} procs={args.processes}")
    with Pool(processes=args.processes) as pool:
        results = pool.map(simulate_symbol, tasks)

    # 결과 병합
    all_trades = pd.concat([r[0] for r in results if not r[0].empty], ignore_index=True) if results else pd.DataFrame()
    all_stats  = pd.concat([r[1] for r in results if not r[1].empty], ignore_index=True) if results else pd.DataFrame()

    trades_path = os.path.join(args.outdir, "bt_tv_entryclose_trades_mp.csv")
    stats_path  = os.path.join(args.outdir, "bt_tv_entryclose_stats_mp.csv")
    all_trades.to_csv(trades_path, index=False)
    all_stats.to_csv(stats_path, index=False)

    # 요약 프린트
    if not all_stats.empty:
        print("\n=== TV Backtest MP (Entry on dip to alert price / Long-only / fees=0.1% roundtrip) ===")
        print(f"Trades saved: {trades_path} (rows={len(all_trades)})")
        print(f"Stats  saved: {stats_path} (rows={len(all_stats)})\n")
        print(all_stats.sort_values("total_net", ascending=False).to_string(index=False))
    else:
        print("No trades generated. Check signal coverage and filters.")

if __name__ == "__main__":
    main()