# backtest_tv_entry_close_mp.py
# -*- coding: utf-8 -*-
"""
TV 알람 백테스트 (멀티프로세싱/캐시/타임존 안정화 포함)

- 엔트리: 알람 발생 시점 가격(근사: 신호 봉의 종가) 이하로 내려오면 진입 (롱)
- 만기: 4h/8h 지정
- 수수료: 왕복 0.1%
- 전략: stable(1.5/1.0), aggressive(2.0/1.25), scalp(1.0/0.75), mid(1.25/1.0), mid(1.75/1.25), tight(0.8/0.8)
- 멀티프로세싱: 심볼 단위 병렬

사용:
python backtest_tv_entry_close_mp.py ./logs/signals_tv.csv --expiry 4h --processes 6 --cache-dir ./cache_tv
"""

import os
import sys
import math
import time
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from sr_engine.data import get_ohlcv

# --------- 유틸 ---------
def ensure_utc_ts(x) -> pd.Timestamp:
    """모든 ts를 UTC-aware 로 표준화"""
    if isinstance(x, pd.Timestamp):
        return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")
    return pd.Timestamp(x, tz="UTC")

def tf_minutes(tf: str) -> int:
    s = tf.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("d"):
        return int(s[:-1]) * 60 * 24
    return 15

def bars_for_hours(hours: int, timeframe: str) -> int:
    return int((hours*60) / tf_minutes(timeframe))

def next_bar_open_idx(ohlcv: pd.DataFrame, sig_ts: pd.Timestamp) -> int:
    """
    sig_ts(UTC) 이후 '다음 봉 오픈 인덱스' 반환.
    numpy datetime64[ns]로 통일해 tz 경고/오류 회피.
    """
    ts = ohlcv["ts"].to_numpy(dtype="datetime64[ns]")
    sig64 = np.datetime64(ensure_utc_ts(sig_ts).to_datetime64())
    return int(np.searchsorted(ts, sig64, side="right"))

def load_signals_tv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 표준 컬럼 강제
    expect = ["ts","event","side","level","touches","symbol","timeframe","extra","source","host","message"]
    miss = [c for c in expect if c not in df.columns]
    if miss:
        # 최소 ts, symbol, message만 있어도 동작하도록 보정
        for c in miss:
            df[c] = ""
    # UTC 통일
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","symbol"]).copy()
    # TV만
    mask_tv = (
        df["source"].fillna("").str.contains(r"(TRADINGVIEW|TV|ALERT|LENOSKY)", case=False, regex=True) |
        df["host"].fillna("").str.contains(r"(\d+\.\d+\.\d+\.\d+|tradingview)", case=False, regex=True) |
        df["message"].fillna("").str.contains(r"(Price in Box|Support Level|Resistance Level|Any alert)", case=False, regex=True)
    )
    tv = df[mask_tv].copy()
    # 심볼 대문자 표준화
    tv["symbol"] = tv["symbol"].astype(str).str.upper()
    # timeframe 비어있으면 기본 15m
    tv["timeframe"] = tv["timeframe"].replace("", "15m")
    return tv

def get_ohlcv_cached(symbol: str, timeframe: str, cache_dir: str="./cache_tv", max_age_min:int=60) -> pd.DataFrame:
    """
    캐시: CSV로 저장(파켓 의존성 제거). 최근 max_age_min 이내면 재사용.
    """
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"ohlcv_{symbol}_{timeframe}.csv")

    use_cache = False
    if os.path.exists(p):
        age_min = (time.time() - os.path.getmtime(p)) / 60.0
        if age_min <= max_age_min:
            use_cache = True

    if use_cache:
        df = pd.read_csv(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        df = get_ohlcv(symbol, timeframe)  # sr_engine.data
        if "ts" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                ts = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
                df = df.reset_index(drop=True)
                df["ts"] = ts
            else:
                raise RuntimeError("OHLCV 데이터에 ts 컬럼이 없습니다.")
        df = df.sort_values("ts").reset_index(drop=True)
        df.to_csv(p, index=False)

    return df[["ts","open","high","low","close","volume"]].copy()

# --------- 엔트리/엑시트 로직 ---------
FEE_RT = 0.001  # 왕복 0.1%

def simulate_symbol(sym: str, rows: pd.DataFrame, timeframe: str, expiry_h: int, tp: float, sl: float) -> list:
    """
    한 심볼에서 주어진 TP/SL/만기로 모든 시그널을 순회, 트레이드 리스트 반환.
    엔트리: 알람 시점가격(근사: 신호 봉의 종가) 이하로 '내려오면' 롱 진입
    """
    ohlcv = get_ohlcv_cached(sym, timeframe)
    ts = ohlcv["ts"].to_numpy(dtype="datetime64[ns]")
    H = ohlcv["high"].to_numpy(float)
    L = ohlcv["low"].to_numpy(float)
    O = ohlcv["open"].to_numpy(float)
    C = ohlcv["close"].to_numpy(float)

    expiry_bars = bars_for_hours(expiry_h, timeframe)
    trades = []

    for _, s in rows.iterrows():
        sig_ts = ensure_utc_ts(s["ts"])
        i_open = next_bar_open_idx(ohlcv, sig_ts)
        if i_open < 1 or i_open >= len(ohlcv):
            continue

        # 알람 시점 가격 근사: 신호 봉 종가
        alert_price = float(C[i_open - 1])

        entered = False
        entry_idx = None
        entry_px = None

        # 엔트리 탐색: alert_price 이하로 내려오는 첫 봉
        for k in range(i_open, min(i_open + expiry_bars, len(ohlcv))):
            if L[k] <= alert_price:  # intrabar 중 alert_price 도달
                # 체결 가격: 봉 시가가 이미 더 낮으면 그 시가, 아니면 alert_price에 체결(리밋 터치)
                entry_px = float(O[k]) if O[k] <= alert_price else alert_price
                entry_idx = k
                entered = True
                break

        if not entered:
            continue  # 미체결 → 스킵

        # 익/손절 라인
        tp_px = entry_px * (1.0 + tp/100.0)
        sl_px = entry_px * (1.0 - sl/100.0)

        exit_idx = None
        exit_px = None
        outcome = "expiry"

        # TP/SL 탐색
        for k in range(entry_idx, min(entry_idx + expiry_bars, len(ohlcv))):
            # 보수적 순서: 저가가 stop 먼저 맞으면 손절, 아니면 고가가 TP 맞으면 익절
            if L[k] <= sl_px:
                exit_idx = k
                # 슬리피지 없이 SL 체결로 가정
                exit_px = sl_px
                outcome = "sl"
                break
            if H[k] >= tp_px:
                exit_idx = k
                exit_px = tp_px
                outcome = "tp"
                break

        # 만기 청산
        if exit_idx is None:
            idx = min(entry_idx + expiry_bars, len(ohlcv)-1)
            exit_idx = idx
            exit_px = float(C[idx])
            outcome = "expiry"

        gross = (exit_px - entry_px) / entry_px
        net = gross - FEE_RT  # 왕복 수수료 차감

        trades.append({
            "symbol": sym,
            "signal_ts": pd.Timestamp(sig_ts).isoformat(),
            "entry_ts": pd.Timestamp(ts[entry_idx]).astype("datetime64[ns]").astype("datetime64[ms]").astype(object).isoformat(),
            "exit_ts": pd.Timestamp(ts[exit_idx]).astype("datetime64[ns]").astype("datetime64[ms]").astype(object).isoformat(),
            "entry_px": round(entry_px, 8),
            "exit_px": round(exit_px, 8),
            "tp_pct": tp,
            "sl_pct": sl,
            "outcome": outcome,
            "gross_pct": round(gross, 6),
            "net_pct": round(net, 6),
        })

    return trades

# --------- 멀티프로세싱 러너 ---------
STRATEGIES = [
    ("stable_1.5/1.0",   1.5, 1.0),
    ("aggressive_2.0/1.25", 2.0, 1.25),
    ("scalp_1.0/0.75",   1.0, 0.75),
    ("mid_1.25/1.0",     1.25, 1.0),
    ("mid_1.75/1.25",    1.75, 1.25),
    ("tight_0.8/0.8",    0.8, 0.8),
]

def worker_one_symbol(args):
    sym, rows, timeframe, expiry_h, tp, sl = args
    return simulate_symbol(sym, rows, timeframe, expiry_h, tp, sl)

def run_mp_for_strategy(tv_df: pd.DataFrame, timeframe: str, expiry_h: int, name: str, tp: float, sl: float, processes: int) -> pd.DataFrame:
    groups = list(tv_df.groupby("symbol"))
    tasks = [(sym, grp, timeframe, expiry_h, tp, sl) for sym, grp in groups]
    if processes <= 0:
        processes = max(1, min(cpu_count()-1, len(tasks)))
    with Pool(processes=processes) as pool:
        results = pool.map(worker_one_symbol, tasks)
    trades = [t for sub in results for t in sub]
    df_tr = pd.DataFrame(trades)
    if not df_tr.empty:
        df_tr.insert(0, "strategy", name)
    else:
        df_tr = pd.DataFrame(columns=["strategy","symbol","signal_ts","entry_ts","exit_ts","entry_px","exit_px","tp_pct","sl_pct","outcome","gross_pct","net_pct"])
    print(f"[BT] {name} done. symbols={tv_df['symbol'].nunique()} trades={len(df_tr)}")
    return df_tr

def agg_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"])
    g = df.groupby("strategy", as_index=False)
    out = g.agg(
        trades=("net_pct","count"),
        win_rate=("net_pct", lambda s: (s > 0).mean() if len(s) else 0.0),
        avg_net=("net_pct","mean"),
        median_net=("net_pct","median"),
        total_net=("net_pct","sum"),
    )
    out["win_rate"] = out["win_rate"].round(4)
    out[["avg_net","median_net","total_net"]] = out[["avg_net","median_net","total_net"]].round(6)
    return out

# --------- 메인 ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals_tv", help="TV 알람 CSV (signals_tv.csv)")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="4h", help="예: 4h or 8h")
    ap.add_argument("--processes", type=int, default=0, help="프로세스 수(0=자동)")
    ap.add_argument("--cache-dir", default="./cache_tv")
    args = ap.parse_args()

    tv = load_signals_tv(args.signals_tv)
    if tv.empty:
        print("TV 신호가 없습니다.")
        return

    # 만기 시간 파싱
    exp = args.expiry.strip().lower()
    if exp.endswith("h"):
        expiry_h = int(exp[:-1])
    else:
        expiry_h = 4

    # 전략별 실행(병렬은 심볼 단위)
    all_trades = []
    for name, tp, sl in STRATEGIES:
        tr = run_mp_for_strategy(tv, args.timeframe, expiry_h, name, tp, sl, args.processes)
        all_trades.append(tr)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    os.makedirs("./logs", exist_ok=True)
    trades_path = "./logs/bt_tv_entryclose_trades.csv"
    stats_path  = "./logs/bt_tv_entryclose_stats.csv"

    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
        stats_df = agg_stats(trades_df)
        stats_df.to_csv(stats_path, index=False)
    else:
        stats_df = pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"])

    print("\n=== TV Backtest (Entry = dip-below-alert price / Long-only / fees=0.1% roundtrip) ===")
    print(f"Trades saved: {trades_path} (rows={len(trades_df)})")
    print(f"Stats  saved: {stats_path} (rows={len(stats_df)})\n")
    if not stats_df.empty:
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(stats_df.sort_values("avg_net", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
