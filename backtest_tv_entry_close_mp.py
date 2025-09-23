# backtest_tv_entry_close_mp.py
# -*- coding: utf-8 -*-
import os, sys, argparse, math
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import timedelta
import numpy as np
import pandas as pd

# sr_engine
from sr_engine.data import get_ohlcv

TV_PATH_DEFAULT = "./logs/signals_tv.csv"
CACHE_DIR = "./logs/cache_tv"
os.makedirs(CACHE_DIR, exist_ok=True)

MAJOR = {"KRW-BTC", "KRW-ETH"}

def ensure_ts_utc(df: pd.DataFrame) -> pd.DataFrame:
    """df['ts']를 UTC tz-aware로 통일하고 오름차순 정렬"""
    df = df.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        ts = None
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        else:
            for cand in ("timestamp", "time", "datetime", "date"):
                if cand in df.columns:
                    ts = pd.to_datetime(df[cand], utc=True, errors="coerce")
                    break
        if ts is None:
            raise RuntimeError("No timestamp column/index to normalize")
        df["ts"] = pd.to_datetime(ts, utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 표준 컬럼 정리
    need = ["ts","symbol","event","side","level","timeframe","message"]
    for c in need:
        if c not in df.columns:
            df[c] = ""
    # ts → UTC aware
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df

def filter_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "major":
        return df[df["symbol"].isin(MAJOR)].copy()
    if group == "alt":
        return df[~df["symbol"].isin(MAJOR)].copy()
    return df

def get_ohlcv_cached(symbol: str, timeframe: str) -> pd.DataFrame:
    # CSV 캐시(파케이로 인한 의존성 회피)
    p = os.path.join(CACHE_DIR, f"{symbol}_{timeframe}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    df = get_ohlcv(symbol, timeframe)
    # ts 보정
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            df = df.reset_index(drop=True)
            df["ts"] = ts
        else:
            df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") if "timestamp" in df.columns else pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df = df.sort_values("ts")
    df.to_csv(p, index=False)
    return df

def next_bar_open_idx(df: pd.DataFrame, sig_ts) -> int:
    """
    sig_ts가 속한 캔들의 다음 캔들 오픈 인덱스를 반환.
    내부 비교는 모두 '나노초 정수' 축에서 수행(충돌 없음).
    """
    df = ensure_ts_utc(df)
    arr_ns = df["ts"].astype("int64").to_numpy()        # UTC ns
    sig_ns = pd.to_datetime(sig_ts, utc=True).value     # UTC ns(int)
    return int(np.searchsorted(arr_ns, sig_ns, side="right"))

def _fees_roundtrip(pct: float) -> float:
    # 왕복 0.1% → -0.001
    return pct - 0.001

def simulate_symbol(args_sym):
    symbol, rows, timeframe, expiries_h = args_sym
    ohlcv = get_ohlcv_cached(symbol, timeframe)
    ohlcv = ohlcv.sort_values("ts").reset_index(drop=True)
    out = []
    for _, s in rows.iterrows():
        sig_ts = pd.Timestamp(s["ts"]).tz_convert("UTC") if pd.Timestamp(s["ts"]).tzinfo else pd.Timestamp(s["ts"]).tz_localize("UTC")
        # 알람 당시 가격 추정: 신호 봉의 close (없으면 그 직전 close)
        i0 = next_bar_open_idx(ohlcv, sig_ts) - 1
        if i0 < 0 or i0 >= len(ohlcv):
            continue
        alert_px = float(ohlcv["close"].iloc[i0])

        # 진입 탐색: i0+1 이후 바에서 'low <= alert_px'
        for expiry_h in expiries_h:
            i_enter = None
            px_enter = None
            for i in range(i0+1, len(ohlcv)):
                low_i = float(ohlcv["low"].iloc[i])
                open_i = float(ohlcv["open"].iloc[i])
                if low_i <= alert_px:
                    i_enter = i
                    px_enter = min(open_i, alert_px)  # 보수적 체결
                    break
            if i_enter is None:
                continue

            # 만기 인덱스(종가 청산)
            dt_enter = pd.Timestamp(ohlcv["ts"].iloc[i_enter]).to_pydatetime()
            dt_expiry = dt_enter + timedelta(hours=expiry_h)
            idx_exp = int(np.searchsorted(ohlcv["ts"].to_numpy(), np.datetime64(pd.Timestamp(dt_expiry, tz="UTC")), side="left"))
            if idx_exp >= len(ohlcv):
                idx_exp = len(ohlcv) - 1
            px_exit = float(ohlcv["close"].iloc[idx_exp])

            ret = (px_exit / px_enter) - 1.0
            ret = _fees_roundtrip(ret)

            out.append({
                "strategy": f"entry<=alert_close_exp{expiry_h}h",
                "symbol": symbol,
                "signal_ts": sig_ts.isoformat(),
                "enter_ts": pd.Timestamp(ohlcv["ts"].iloc[i_enter]).isoformat(),
                "exit_ts": pd.Timestamp(ohlcv["ts"].iloc[idx_exp]).isoformat(),
                "enter_px": px_enter,
                "exit_px": px_exit,
                "net": ret,
                "expiry_h": expiry_h,
            })
    return pd.DataFrame(out)

def agg_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["strategy","trades","win_rate","avg_net","median_net","total_net"])
    g = df.groupby("strategy")
    res = g["net"].agg(["count","mean","median","sum"]).reset_index()
    res.columns = ["strategy","trades","avg_net","median_net","total_net"]
    res["win_rate"] = (df["net"] > 0).groupby(df["strategy"]).mean().values
    return res[["strategy","trades","win_rate","avg_net","median_net","total_net"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", nargs="?", default=TV_PATH_DEFAULT)
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="4h,8h", help="comma: e.g. 4h,8h")
    ap.add_argument("--group", default="all", choices=["all","major","alt"])
    ap.add_argument("--symbols", default="", help="comma list to force symbols")
    ap.add_argument("--procs", type=int, default=max(1, cpu_count()-1))
    args = ap.parse_args()

    df = load_signals(args.signals)
    df = filter_group(df, args.group)
    if args.symbols:
        want = set([x.strip().upper() for x in args.symbols.split(",") if x.strip()])
        df = df[df["symbol"].isin(want)].copy()

    expiries_h = []
    for chunk in args.expiry.split(","):
        chunk = chunk.strip().lower()
        if chunk.endswith("h"):
            expiries_h.append(int(chunk[:-1]))
    if not expiries_h:
        expiries_h = [4,8]

    # 심볼별 묶음
    tasks = []
    for sym, rows in df.groupby("symbol"):
        tasks.append((sym, rows.sort_values("ts"), args.timeframe, expiries_h))

    with Pool(processes=args.procs) as pool:
        parts = pool.map(simulate_symbol, tasks)
    trades = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    stats = agg_stats(trades)

    os.makedirs("./logs", exist_ok=True)
    trades_path = "./logs/bt_tv_entryclose_mp_trades.csv"
    stats_path  = "./logs/bt_tv_entryclose_mp_stats.csv"
    trades.to_csv(trades_path, index=False)
    stats.to_csv(stats_path, index=False)

    print(f"[BT][TV-MP] Trades: {len(trades)} → {trades_path}")
    print(f"[BT][TV-MP] Stats : {len(stats)} → {stats_path}")
    if not stats.empty:
        print("\n=== TV-MP Backtest Summary ===")
        print(stats.sort_values(["avg_net","win_rate"], ascending=[False, False]).to_string(index=False))

if __name__ == "__main__":
    main()