# -*- coding: utf-8 -*-
"""
backtest_tv_entry_close_mp.py

멀티프로세싱 TV 신호 백테스트 (Entry=signal 이후 가격이 signal_price 이하일 때 진입)
- Exit: TP/SL or 만기(종가청산)
- 수수료: 왕복 0.1%
- Long-only
- 안정성 패치:
  * pyupbit None 응답 재시도/백오프 + 안전 처리
  * tz-naive/aware 섞임 정규화
  * 심볼별 OHLCV 캐시(프로세스 내)

예)
  python backtest_tv_entry_close_mp.py .\logs\signals_tv.csv --group alt --expiry 4h,8h --procs 20
"""

import os
import argparse
import time
import multiprocessing as mp
from datetime import timedelta

import numpy as np
import pandas as pd
import pyupbit

# -------------------- 공통 유틸 --------------------
def _to_utc(tslike) -> pd.Timestamp:
    """tz-naive/aware 섞임을 안전하게 UTC로 정규화."""
    ts = pd.Timestamp(tslike)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise RuntimeError("signals file must have 'ts' column")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "symbol", "event"])
    return df

# -------------------- OHLCV 로드 (재시도/캐시) --------------------
_OHLCV_CACHE = {}

def get_ohlcv_safe(symbol: str, tf: str, count: int = 2000, max_retries: int = 6) -> pd.DataFrame:
    """pyupbit None 응답 대응 + 백오프 재시도."""
    key = (symbol, tf, count)
    if key in _OHLCV_CACHE:
        return _OHLCV_CACHE[key]

    backoff = 1.0
    last_err = None
    for _ in range(max_retries):
        try:
            df = pyupbit.get_ohlcv(ticker=symbol, interval=tf, count=count)
            if df is not None and len(df) > 0:
                df = df.reset_index().rename(columns={"index": "ts"})
                # pyupbit index는 tz 정보가 없는 Timestamp일 수 있음 → UTC로 통일
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                # 컬럼 존재 보정
                for col in ["open","high","low","close"]:
                    if col not in df.columns:
                        raise RuntimeError(f"missing column '{col}'")
                _OHLCV_CACHE[key] = df
                return df
        except Exception as e:
            last_err = e
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 5.0)

    # 최종 실패 시 빈 DF 반환(상위에서 스킵)
    if last_err:
        print(f"[{symbol}] OHLCV load error (exhausted): {repr(last_err)}")
    return pd.DataFrame(columns=["ts","open","high","low","close","volume","value"])

# -------------------- 인덱싱 유틸 --------------------
def next_bar_open_idx(df: pd.DataFrame, sig_ts: pd.Timestamp) -> int:
    """신호 발생 직후 첫 봉 오픈 인덱스(right)"""
    if len(df) == 0:
        return 0
    ts_np = df["ts"].to_numpy("datetime64[ns]")  # numpy naive
    sig_ts_utc = _to_utc(sig_ts)
    return int(np.searchsorted(ts_np, sig_ts_utc.to_datetime64(), side="right"))

# -------------------- 개별 심볼 시뮬 --------------------
def simulate_symbol(task):
    sym, rows, timeframe, tp, sl, expiry_h = task

    # OHLCV 로드
    ohlcv = get_ohlcv_safe(sym, timeframe)
    if len(ohlcv) == 0:
        # 실패 시 빈 DF (스키마 유지)
        return pd.DataFrame(columns=[
            "symbol","sig_ts","sig_price","entry_ts","entry_price",
            "exit_ts","exit_price","exit_reason","tp","sl","expiry_h","net"
        ])

    trades = []
    ts_np = ohlcv["ts"].to_numpy("datetime64[ns]")

    for _, s in rows.iterrows():
        # ts를 안전하게 UTC로
        sig_ts = _to_utc(s["ts"])

        # signal price(레벨) 결정: level / sig_price 컬럼 우선
        sp = s.get("level", np.nan)
        if pd.isna(sp):
            sp = s.get("sig_price", np.nan)
        if pd.isna(sp):
            # message에서 center=.. 패턴을 파싱하는 식으로 확장 가능
            continue
        sig_price = float(sp)

        # 신호 직후 봉부터 탐색
        i0 = next_bar_open_idx(ohlcv, sig_ts) - 1
        if i0 < 0:
            i0 = 0
        if i0 >= len(ohlcv) - 1:
            continue

        # 만기 위치
        expiry_ts = sig_ts + timedelta(hours=expiry_h)
        idx_exp = int(np.searchsorted(ts_np, _to_utc(expiry_ts).to_datetime64(), side="left"))
        if idx_exp <= i0 + 1:
            continue
        idx_exp = min(idx_exp, len(ohlcv) - 1)

        # 엔트리: 이후 low가 signal_price 이하가 되는 첫 봉
        entry_ts = entry_price = None
        for i in range(i0 + 1, idx_exp + 1):
            if float(ohlcv["low"].iloc[i]) <= sig_price:
                entry_ts = ohlcv["ts"].iloc[i]
                entry_price = sig_price
                entry_i = i
                break
        if entry_price is None:
            continue

        tp_price = entry_price * (1 + tp/100.0)
        sl_price = entry_price * (1 - sl/100.0)

        exit_ts = exit_price = exit_reason = None
        for j in range(entry_i, idx_exp + 1):
            high_j = float(ohlcv["high"].iloc[j])
            low_j  = float(ohlcv["low"].iloc[j])
            close_j = float(ohlcv["close"].iloc[j])
            ts_j = ohlcv["ts"].iloc[j]

            if high_j >= tp_price:
                exit_ts, exit_price, exit_reason = ts_j, tp_price, "tp"
                break
            if low_j <= sl_price:
                exit_ts, exit_price, exit_reason = ts_j, sl_price, "sl"
                break

        if exit_ts is None:
            # 만기 종가 청산
            exit_ts = ohlcv["ts"].iloc[idx_exp]
            exit_price = float(ohlcv["close"].iloc[idx_exp])
            exit_reason = "expiry"

        fee = 0.001  # round-trip 0.1%
        net = (exit_price / entry_price - 1.0) - fee

        trades.append({
            "symbol": sym, "sig_ts": sig_ts, "sig_price": sig_price,
            "entry_ts": entry_ts, "entry_price": entry_price,
            "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": exit_reason,
            "tp": tp, "sl": sl, "expiry_h": expiry_h, "net": net
        })

    if not trades:
        return pd.DataFrame(columns=[
            "symbol","sig_ts","sig_price","entry_ts","entry_price",
            "exit_ts","exit_price","exit_reason","tp","sl","expiry_h","net"
        ])
    return pd.DataFrame(trades)

# -------------------- 통계 --------------------
def agg_stats(df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "trades": len(df),
        "win_rate": (df["net"] > 0).mean() if len(df) else 0.0,
        "avg_net": df["net"].mean() if len(df) else 0.0,
        "median_net": df["net"].median() if len(df) else 0.0,
        "total_net": df["net"].sum() if len(df) else 0.0,
    })

# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="4h,8h")
    ap.add_argument("--group", default="all", choices=["all","major","alt"])
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    df_sig = load_signals(args.signals)

    # 심볼 그룹 필터
    majors = ["KRW-BTC", "KRW-ETH"]
    if args.group == "major":
        df_sig = df_sig[df_sig["symbol"].isin(majors)]
    elif args.group == "alt":
        df_sig = df_sig[~df_sig["symbol"].isin(majors)]

    # 만기/전략 세트
    EXPIRIES = [int(x.strip().replace("h","")) for x in args.expiry.split(",") if x.strip()]
    STRATS = [
        ("stable",     1.5, 1.0),
        ("aggressive", 2.0, 1.25),
        ("scalp",      1.0, 0.75),
        ("mid",        1.25, 1.0),
        ("mid2",       1.75, 1.25),
        ("tight",      0.8, 0.8),
    ]

    # 태스크 생성: (심볼, 행들, tf, tp, sl, 만기)
    tasks = []
    for sym, rows in df_sig.groupby("symbol"):
        for _, tp, sl in STRATS:
            for expiry_h in EXPIRIES:
                tasks.append((sym, rows, args.timeframe, tp, sl, expiry_h))

    # 병렬 실행
    with mp.Pool(processes=args.procs) as pool:
        parts = pool.map(simulate_symbol, tasks)

    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[
        "symbol","sig_ts","sig_price","entry_ts","entry_price",
        "exit_ts","exit_price","exit_reason","tp","sl","expiry_h","net"
    ])

    # 그룹바이 키들이 항상 존재하도록 보장됨(simulate_symbol에서 생성)
    stats = trades.groupby(["tp","sl","expiry_h"], as_index=False, dropna=False).apply(agg_stats)

    os.makedirs("./logs", exist_ok=True)
    out_trades = "./logs/bt_tv_entryclose_trades.csv"
    out_stats  = "./logs/bt_tv_entryclose_stats.csv"
    trades.to_csv(out_trades, index=False)
    stats.to_csv(out_stats, index=False)

    print("\n=== TV Backtest (Entry=price<=signal_price / Long-only / fees=0.1% RT) ===")
    print(f"Trades saved: {out_trades} (rows={len(trades)})")
    print(f"Stats  saved: {out_stats} (rows={len(stats)})")
    print(stats)

if __name__ == "__main__":
    main()