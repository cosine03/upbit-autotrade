# -*- coding: utf-8 -*-
"""
TV signals 4-event backtest (multiprocessing, close-entry)

- events: levelX_detected / price_in_box / box_breakout / line_breakout
- long-only: resistance만 사용 (support는 스킵)
- entry: 신호가 뜬 봉의 다음 봉 '시가'로 진입 (close-entry 스타일)
- exit: TP/SL(%) 또는 만기(복수 가능: 4h,8h 등) 중 먼저 도달
- fees: round-trip 비율(기본 0.1% = 0.001)
- timeframe: OHLCV 주기 (기본 15m)

Usage:
  python backtest_tv_events_mp.py ./logs/signals_tv.csv \
      --timeframe 15m --expiries 4h,8h --tp 1.5 --sl 1.0 --fee 0.001 --procs 24

Outputs:
  ./logs/bt_tv_events_trades.csv
  ./logs/bt_tv_events_stats.csv
  (이벤트군별 분리 저장도 수행)
"""

import os
import sys
import math
import argparse
import warnings
from functools import lru_cache
from multiprocessing import Pool, get_context, cpu_count

import numpy as np
import pandas as pd

# 최소 의존: sr_engine.data.get_ohlcv 사용
from sr_engine.data import get_ohlcv

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------- UTILS (TZ & IO) --------------

def ensure_logs_dir():
    outdir = "./logs"
    os.makedirs(outdir, exist_ok=True)
    return outdir

def parse_expiries(arg: str):
    out = []
    for tok in arg.split(","):
        tok = tok.strip().lower()
        if not tok: 
            continue
        # '4h', '8h', '24h' ...
        if tok.endswith("h"):
            out.append(float(tok[:-1]))
        else:
            out.append(float(tok))
    return out

def to_utc_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """df에 tz-aware UTC 'ts' 컬럼 보장."""
    if "ts" in df.columns:
        s = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.copy()
        df["ts"] = s
        return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    # index로 있는 경우
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        df = df.copy()
        df["ts"] = ts
        return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    # 후보 찾기
    for c in ("timestamp", "time", "datetime", "date"):
        if c in df.columns:
            s = pd.to_datetime(df[c], utc=True, errors="coerce")
            df = df.copy()
            df["ts"] = s
            return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    # 실패하면 스킵
    return df

def ts_series_to_ns(s: pd.Series) -> np.ndarray:
    """tz-aware(series)-> numpy datetime64[ns] array"""
    # pandas가 tz-aware를 바로 astype 못하므로 tz 제거 후 ns로
    s = pd.to_datetime(s, utc=True, errors="coerce")
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]").to_numpy()

def ts_to_ns_scalar(ts: pd.Timestamp) -> np.datetime64:
    ts = pd.to_datetime(ts, utc=True)
    ts = ts.tz_convert("UTC")
    return np.datetime64(ts.tz_localize(None).to_pydatetime())

def minutes_of(tf: str) -> int:
    t = tf.strip().lower()
    if t.endswith("m"):
        return int(t[:-1])
    if t.endswith("h"):
        return int(t[:-1]) * 60
    if t.endswith("d"):
        return int(t[:-1]) * 1440
    return 15

# -------------- OHLCV CACHE --------------

# --- replace this function in backtest_tv_events_mp.py ---

from pandas.api.types import is_datetime64_any_dtype

@lru_cache(maxsize=256)
def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    df = get_ohlcv(symbol, timeframe)
    if df is None or len(df) == 0:
        print(f"[{symbol}] get_ohlcv returned empty.")
        return pd.DataFrame()

    # 1) DatetimeIndex를 바로 ts로 사용 (가장 안전)
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        df2 = df.copy()
        df2["ts"] = ts
        df2 = df2.reset_index(drop=True)
    else:
        # 2) reset_index 후, datetime형 컬럼을 찾아 'ts'로 표준화
        df2 = df.reset_index(drop=False)

        ts_col = None
        # 2-a) 이미 datetime dtype인 컬럼이 있으면 우선 채택
        for c in df2.columns:
            if is_datetime64_any_dtype(df2[c]):
                ts_col = c
                break

        # 2-b) 없다면 후보 이름을 파싱해서 생성
        if ts_col is None:
            for cand in ("ts", "index", "timestamp", "time", "datetime", "date"):
                if cand in df2.columns:
                    parsed = pd.to_datetime(df2[cand], utc=True, errors="coerce")
                    if parsed.notna().any():
                        df2["ts"] = parsed
                        ts_col = "ts"
                        break

        # 2-c) 그래도 못 찾으면 포기(스킵)
        if ts_col is None and "ts" not in df2.columns:
            print(f"[{symbol}] OHLCV missing datetime column; skipping.")
            return pd.DataFrame()

        if "ts" not in df2.columns:
            # datetime dtype 컬럼명을 ts로 바꾸고 UTC 정규화
            df2 = df2.rename(columns={ts_col: "ts"})
            df2["ts"] = pd.to_datetime(df2["ts"], utc=True, errors="coerce")

    # 3) 필요한 컬럼만 남기고 정렬/정리
    keep = [c for c in ("ts", "open", "high", "low", "close") if c in df2.columns]
    if "ts" not in keep:
        print(f"[{symbol}] OHLCV still has no ts after normalization; skipping.")
        return pd.DataFrame()

    df2 = (df2[keep]
           .dropna(subset=["ts", "open", "high", "low", "close"])
           .sort_values("ts")
           .reset_index(drop=True))
    return df2

# -------------- ENTRY/EXIT LOGIC --------------

def next_bar_open_idx(ohlcv: pd.DataFrame, sig_ts: pd.Timestamp) -> int:
    """
    신호가 뜬 봉의 '다음 봉 시가' 인덱스 반환.
    """
    if ohlcv.empty:
        return -1
    ts_arr = ts_series_to_ns(ohlcv["ts"])
    key = ts_to_ns_scalar(sig_ts)
    # sig_ts가 속한 봉의 다음 봉 시작(= searchsorted right)
    i = int(np.searchsorted(ts_arr, key, side="right"))
    return i if 0 <= i < len(ohlcv) else -1

def bars_for_hours(timeframe: str, hours: float) -> int:
    m = minutes_of(timeframe)
    return max(1, int(round(hours * 60.0 / m)))

def simulate_symbol(symbol: str,
                    rows: pd.DataFrame,
                    timeframe: str,
                    tp_pct: float,
                    sl_pct: float,
                    fee_rt: float,
                    expiries_h: list) -> pd.DataFrame:
    """
    하나의 symbol에 대해 모든 만기 조합을 계산해 trades DataFrame 반환.
    롱온리: side=='resistance'만 사용.
    """
    ohlcv = load_ohlcv(symbol, timeframe)
    if ohlcv.empty:
        return pd.DataFrame(columns=[
            "symbol","event","touches","expiry_h","ts_entry","ts_exit",
            "entry","exit","tp","sl","gross_r","net_r"
        ])

    ts = ohlcv["ts"].to_list()  # pandas Timestamp(UTC)
    prices_o = ohlcv["open"].to_numpy()
    highs = ohlcv["high"].to_numpy()
    lows = ohlcv["low"].to_numpy()
    closes = ohlcv["close"].to_numpy()

    out = []
    for s in rows.itertuples():
        # 필수 컬럼 가드
        if pd.isna(s.ts) or pd.isna(s.side) or pd.isna(s.event):
            continue
        # 롱온리: resistance만
        if str(s.side).lower() != "resistance":
            continue

        sig_ts = pd.to_datetime(s.ts, utc=True, errors="coerce")
        if pd.isna(sig_ts):
            continue

        i_entry = next_bar_open_idx(ohlcv, sig_ts)
        if i_entry < 0:
            continue

        ep = float(prices_o[i_entry])
        tp_line = ep * (1.0 + tp_pct/100.0)
        sl_line = ep * (1.0 - sl_pct/100.0)

        for eh in expiries_h:
            ebar = i_entry + bars_for_hours(timeframe, eh)
            ebar = min(ebar, len(ohlcv)-1)

            hit_tp = hit_sl = -1
            # 진입 다음 봉부터 만기 봉까지 검사
            for k in range(i_entry+1, ebar+1):
                if highs[k] >= tp_line and hit_tp < 0:
                    hit_tp = k
                if lows[k] <= sl_line and hit_sl < 0:
                    hit_sl = k
                if hit_tp > 0 or hit_sl > 0:
                    break

            if hit_tp > 0 and (hit_sl < 0 or hit_tp <= hit_sl):
                x_idx = hit_tp
                x_px = tp_line
                gross_r = (x_px/ep) - 1.0
            elif hit_sl > 0:
                x_idx = hit_sl
                x_px = sl_line
                gross_r = (x_px/ep) - 1.0
            else:
                x_idx = ebar
                x_px = float(closes[x_idx])
                gross_r = (x_px/ep) - 1.0

            # 수수료: 왕복 fee_rt 차감 (예: 0.001)
            net_r = gross_r - fee_rt

            out.append({
                "symbol": symbol,
                "event": s.event,
                "touches": s.touches,
                "expiry_h": eh,
                "ts_entry": ts[i_entry],
                "ts_exit": ts[x_idx],
                "entry": ep,
                "exit": x_px,
                "tp": tp_pct,
                "sl": sl_pct,
                "gross_r": gross_r,
                "net_r": net_r,
            })

    return pd.DataFrame(out)

# -------------- DRIVER (MP) --------------

EVENT_GROUPS = {
    "detected": lambda e: str(e).startswith("level"),
    "price_in_box": lambda e: str(e).lower() == "price_in_box",
    "box_breakout": lambda e: str(e).lower() == "box_breakout",
    "line_breakout": lambda e: str(e).lower() == "line_breakout",
}

def run_group(df_sig: pd.DataFrame,
              group_name: str,
              timeframe: str,
              tp: float,
              sl: float,
              fee_rt: float,
              expiries_h: list,
              procs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    이벤트 그룹 하나(예: 'line_breakout')에 대해 멀티프로세싱 실행.
    """
    filt = EVENT_GROUPS[group_name]
    gdf = df_sig[df_sig["event"].apply(filt)].copy()
    if gdf.empty:
        return pd.DataFrame(), pd.DataFrame()

    symbols = sorted(gdf["symbol"].dropna().unique())
    tasks = []
    for sym in symbols:
        rows = gdf[gdf["symbol"] == sym].copy()
        rows["ts"] = pd.to_datetime(rows["ts"], utc=True, errors="coerce")
        rows = rows.dropna(subset=["ts"])
        if rows.empty:
            continue
        tasks.append((sym, rows, timeframe, tp, sl, fee_rt, tuple(expiries_h)))

    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    print(f"[BT][{group_name}] start: symbols={len(symbols)} rows={len(gdf)} tasks={len(tasks)} procs={procs}")

    # spawn 컨텍스트로 pickle 문제 회피
    ctx = get_context("spawn")
    with ctx.Pool(processes=procs) as pool:
        parts = pool.starmap(simulate_symbol, tasks)

    trades = pd.concat([p for p in parts if p is not None and not p.empty],
                       ignore_index=True) if parts else pd.DataFrame()

    def agg_stats(x: pd.DataFrame):
        if x.empty:
            return pd.Series({"trades": 0, "win_rate": 0.0, "avg_net": 0.0, "median_net": 0.0, "total_net": 0.0})
        wins = (x["net_r"] > 0).mean()
        return pd.Series({
            "trades": len(x),
            "win_rate": float(wins),
            "avg_net": float(x["net_r"].mean()),
            "median_net": float(x["net_r"].median()),
            "total_net": float(x["net_r"].sum()),
        })

    stats = (trades.groupby(["expiry_h"], as_index=False)
                   .apply(agg_stats)
                   .assign(group=group_name)
                   .loc[:, ["group","expiry_h","trades","win_rate","avg_net","median_net","total_net"]]
                   .reset_index(drop=True)
             ) if not trades.empty else pd.DataFrame(columns=["group","expiry_h","trades","win_rate","avg_net","median_net","total_net"])

    return trades, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h", help="comma separated, e.g. 4h,8h")
    ap.add_argument("--tp", type=float, default=1.5, help="take-profit percent")
    ap.add_argument("--sl", type=float, default=1.0, help="stop-loss percent")
    ap.add_argument("--fee", type=float, default=0.001, help="round-trip fee rate (e.g. 0.001 = 0.1%)")
    ap.add_argument("--procs", type=int, default=max(1, min(cpu_count()-1, 16)))
    args = ap.parse_args()

    expiries_h = parse_expiries(args.expiries)
    outdir = ensure_logs_dir()

    # --- load signals
    df = pd.read_csv(args.signals)
    # 표준 컬럼 보정
    need_cols = ["ts","event","side","touches","symbol"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[need_cols].copy()

    # UTC normalize
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","event","side","symbol"]).reset_index(drop=True)

    print(f"[BT] signals rows={len(df)}, symbols={df['symbol'].nunique()}, timeframe={args.timeframe}")

    all_trades = []
    all_stats = []

    for grp in ("detected","price_in_box","box_breakout","line_breakout"):
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl, args.fee, expiries_h, args.procs)
        # 이벤트군별 저장
        if not tr.empty:
            tr_path = os.path.join(outdir, f"bt_tv_events_trades_{grp}.csv")
            tr.to_csv(tr_path, index=False)
            print(f"[BT][{grp}] trades -> {tr_path} (rows={len(tr)})")
        if not st.empty:
            st_path = os.path.join(outdir, f"bt_tv_events_stats_{grp}.csv")
            st.to_csv(st_path, index=False)
            print(f"[BT][{grp}] stats  -> {st_path} (rows={len(st)})")

        if not tr.empty: all_trades.append(tr.assign(group=grp))
        if not st.empty: all_stats.append(st)

    trades = (pd.concat(all_trades, ignore_index=True)
              if all_trades else pd.DataFrame())
    stats  = (pd.concat(all_stats, ignore_index=True)
              if all_stats else pd.DataFrame())

    # 전체 합본 저장
    trades_path = os.path.join(outdir, "bt_tv_events_trades.csv")
    stats_path  = os.path.join(outdir, "bt_tv_events_stats.csv")
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
    if not stats.empty:
        # 보기 좋게 정렬: group -> expiry_h
        stats = stats.sort_values(["group","expiry_h"]).reset_index(drop=True)
        stats.to_csv(stats_path, index=False)

    # 콘솔 요약
    print("\n=== Summary (by event group & expiry) ===")
    if stats.empty:
        print("(no trades)")
    else:
        print(stats.to_string(index=False))

    if trades.empty:
        print("\n[BT] No trades generated.")
    else:
        print(f"\n[BT] saved -> {trades_path} (rows={len(trades)})")
        print(f"[BT] saved -> {stats_path} (rows={len(stats)})")

if __name__ == "__main__":
    main()