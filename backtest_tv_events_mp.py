# -*- coding: utf-8 -*-
"""
backtest_tv_events_mp.py
- TradingView 신호(logs/signals_tv_enriched.csv 등)를 읽어, 로컬 OHLCV(CSV)로 멀티프로세싱 백테스트
- 외부 의존성 없음 (로컬 csv만 사용)
- 경고 제거(Series.view -> astype, groupby include_groups=False), 요약표 출력

사용 예:
  python -u backtest_tv_events_mp.py .\logs\signals_tv_enriched.csv ^
    --timeframe 15m --expiries 0.5h,1h,2h ^
    --tp 1.5 --sl 0.8 --fee 0.001 ^
    --dist-max 0.02 ^
    --procs 24 ^
    --ohlcv-roots ".;.\data;.\data\ohlcv;.\ohlcv;.\logs;.\logs\ohlcv" ^
    --ohlcv-patterns "data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv" ^
    --assume-ohlcv-tz UTC ^
    --outdir .\logs\bt_tv_dm0020_tp1p5_sl0p8
"""

import os
import argparse
import itertools
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


BANNER = "[BT] BACKTEST TV EVENTS (LOCAL_SIM v3, no external deps)"


# -------------------------- IO / Utility --------------------------

def to_utc_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def series_to_ns_utc(s: pd.Series) -> np.ndarray:
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.astype("int64").to_numpy()  # FutureWarning-safe

def ensure_ts_col(df: pd.DataFrame, assume_tz: str = "UTC") -> pd.DataFrame:
    out = df.copy()
    if "ts" in out.columns:
        ts = pd.to_datetime(out["ts"], errors="coerce", utc=(assume_tz.upper() == "UTC"))
        if ts.dt.tz is None:
            # local naive -> localize then convert
            ts = ts.dt.tz_localize(assume_tz).dt.tz_convert("UTC")
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("OHLCV must have DatetimeIndex or 'ts' column")
        ts = out.index
        if ts.tz is None:
            ts = ts.tz_localize(assume_tz).tz_convert("UTC")
        else:
            ts = ts.tz_convert("UTC")
    out["ts"] = ts
    return out.reset_index(drop=True)

def load_csv(path: str, assume_tz: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # 기대 컬럼: ts/open/high/low/close/volume
        # ts를 문자열일 수 있으므로 ensure_ts_col에서 처리
        need = {"open", "high", "low", "close"}
        if not need.issubset(set(c.lower() for c in df.columns)):
            # 대소문자 보정
            rename = {c: c.lower() for c in df.columns}
            df.rename(columns=rename, inplace=True)
        df = ensure_ts_col(df, assume_tz=assume_tz)
        keep = ["ts", "open", "high", "low", "close", "volume"] if "volume" in df.columns else ["ts", "open", "high", "low", "close"]
        df = df[keep].dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
        return df
    except Exception:
        return None

def get_ohlcv_csv(symbol: str,
                  timeframe: str,
                  roots: List[str],
                  patterns: List[str],
                  assume_tz: str = "UTC") -> Optional[pd.DataFrame]:
    # 패턴 치환 및 검색
    tried = []
    for root in roots:
        root = root.strip()
        for pat in patterns:
            pat = pat.strip()
            rel = pat.format(symbol=symbol, timeframe=timeframe)
            if not (rel.lower().endswith(".csv")):
                rel += ".csv"
            path = os.path.join(root, rel) if root else rel
            path = os.path.normpath(path)
            tried.append(path)
            df = load_csv(path, assume_tz=assume_tz)
            if df is not None and not df.empty:
                print(f"[BT][LOCAL] {symbol} {timeframe} -> {path} (rows={len(df)})")
                return df
    # 못 찾으면 None
    print(f"[BT][WARN] {symbol} {timeframe}: OHLCV not found in any of:\n  - " + "\n  - ".join(tried[:6]) + ("..." if len(tried) > 6 else ""))
    return None


# -------------------------- Core Simulation --------------------------

def find_entry_index(ohlcv: pd.DataFrame, sig_ts_utc: pd.Timestamp) -> Optional[int]:
    ts64 = series_to_ns_utc(ohlcv["ts"])
    key = np.int64(sig_ts_utc.value)
    i = int(np.searchsorted(ts64, key, side="right")) - 1
    if i < 0 or i >= len(ohlcv):
        return None
    return i

def expiry_to_bars(expiry_h: float, timeframe: str) -> int:
    tf = timeframe.lower().strip()
    if tf.endswith("m"):
        step_m = int(tf[:-1])
    elif tf.endswith("h"):
        step_m = int(tf[:-1]) * 60
    elif tf.endswith("d"):
        step_m = int(tf[:-1]) * 60 * 24
    else:
        step_m = 15
    bars = int(round((expiry_h * 60) / step_m))
    return max(1, bars)

def simulate_one_trade(ohlcv: pd.DataFrame,
                       i_entry: int,
                       expiry_bars: int,
                       side_used: str,
                       tp_pct: float,
                       sl_pct: float,
                       fee: float) -> Dict:
    # entry 기준가: 직전/해당 봉의 종가
    entry = float(ohlcv["close"].iloc[i_entry])
    entry_ts = ohlcv["ts"].iloc[i_entry]
    # TP/SL 레벨
    if side_used == "resistance":  # short
        tp_level = entry * (1 - tp_pct / 100.0)
        sl_level = entry * (1 + sl_pct / 100.0)
    else:  # default long
        tp_level = entry * (1 + tp_pct / 100.0)
        sl_level = entry * (1 - sl_pct / 100.0)

    # 앞쪽 구간 탐색
    i_end = min(len(ohlcv) - 1, i_entry + expiry_bars)
    hit = None
    exit_px = None
    exit_ts = None

    for j in range(i_entry + 1, i_end + 1):
        hi = float(ohlcv["high"].iloc[j])
        lo = float(ohlcv["low"].iloc[j])
        tsj = ohlcv["ts"].iloc[j]
        if side_used == "resistance":  # short
            # TP 먼저?
            if lo <= tp_level:
                hit = "TP"
                exit_px = tp_level
                exit_ts = tsj
                break
            # SL?
            if hi >= sl_level:
                hit = "SL"
                exit_px = sl_level
                exit_ts = tsj
                break
        else:  # long
            if hi >= tp_level:
                hit = "TP"
                exit_px = tp_level
                exit_ts = tsj
                break
            if lo <= sl_level:
                hit = "SL"
                exit_px = sl_level
                exit_ts = tsj
                break

    if hit is None:
        # 만기종료: 마지막 종가
        exit_px = float(ohlcv["close"].iloc[i_end])
        exit_ts = ohlcv["ts"].iloc[i_end]
        hit = "EXP"

    # 손익 계산 (왕복 수수료 2*fee 적용)
    if side_used == "resistance":
        gross = (entry / exit_px) - 1.0
    else:
        gross = (exit_px / entry) - 1.0
    net = gross - (2.0 * fee)

    return {
        "ts_open": entry_ts,
        "ts_close": exit_ts,
        "entry": entry,
        "exit": exit_px,
        "result": hit,
        "gross": gross,
        "net": net,
    }

def parse_side_from_row(row: pd.Series) -> str:
    s = str(row.get("side", "") or "").strip().lower()
    if s in ("support", "resistance"):
        return s
    msg = str(row.get("message", "") or "")
    if pd.notna(msg):
        msg_l = msg.lower()
        if "resistance" in msg_l:
            return "resistance"
        if "support" in msg_l:
            return "support"
    return "support"  # default long

def parse_expiries(expiries: str) -> List[float]:
    out = []
    for x in expiries.split(","):
        x = x.strip().lower()
        if x.endswith("h"):
            out.append(float(x[:-1]))
        elif x.endswith("m"):
            out.append(float(x[:-1]) / 60.0)
        else:
            # plain number: hours
            out.append(float(x))
    return out

# -------------------------- Worker --------------------------

def simulate_symbol(symbol: str,
                    df_sig: pd.DataFrame,
                    timeframe: str,
                    tp: float, sl: float, fee: float,
                    expiries_h: List[float],
                    roots: List[str],
                    patterns: List[str],
                    assume_tz: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 로컬 OHLCV
    ohlcv = get_ohlcv_csv(symbol, timeframe, roots=roots, patterns=patterns, assume_tz=assume_tz)
    if ohlcv is None or ohlcv.empty:
        print(f"[{symbol}] get_ohlcv returned empty.")
        return pd.DataFrame(), pd.DataFrame()

    trades = []
    stats_rows = []

    for _, r in df_sig.iterrows():
        side_used = parse_side_from_row(r)
        sig_ts = to_utc_ts(r["ts"])
        i_entry = find_entry_index(ohlcv, sig_ts)
        if i_entry is None:
            continue

        for ex in expiries_h:
            bars = expiry_to_bars(ex, timeframe)
            sim = simulate_one_trade(ohlcv, i_entry, bars, side_used, tp, sl, fee)
            trades.append({
                "event": str(r.get("event", "")),
                "symbol": symbol,
                "side_used": side_used,
                "expiry_h": ex,
                "ts_open": sim["ts_open"],
                "ts_close": sim["ts_close"],
                "entry": sim["entry"],
                "exit": sim["exit"],
                "result": sim["result"],
                "gross": sim["gross"],
                "net": sim["net"],
            })

    if trades:
        tdf = pd.DataFrame(trades)
        # 이벤트별 요약 (심볼 단위)
        sdf = tdf.groupby(["event", "expiry_h"], include_groups=False)["net"].agg(
            trades="count",
            win_rate=lambda x: (x > 0).mean(),
            avg_net="mean",
            median_net="median",
            total_net="sum",
        ).reset_index()
        return tdf, sdf
    else:
        return pd.DataFrame(), pd.DataFrame()

# -------------------------- Group Runner --------------------------

def run_group(df: pd.DataFrame,
              group_name: str,
              timeframe: str,
              tp: float, sl: float, fee: float,
              expiries_h: List[float],
              procs: int,
              roots: List[str],
              patterns: List[str],
              assume_tz: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["event"] == group_name].copy()
    if sub.empty:
        print(f"[BT][{group_name}] no tasks.")
        return pd.DataFrame(), pd.DataFrame()

    symbols = sorted(sub["symbol"].dropna().astype(str).unique())
    tasks = []
    for sym in symbols:
        df_sym = sub[sub["symbol"] == sym].copy().reset_index(drop=True)
        tasks.append((sym, df_sym, timeframe, tp, sl, fee, expiries_h, roots, patterns, assume_tz))

    print(f"[BT][{group_name}] start: symbols={len(symbols)} rows={len(sub)} tasks={len(tasks)} procs={procs}")

    if procs <= 1:
        parts = [simulate_symbol(*t) for t in tasks]
    else:
        with Pool(processes=min(procs, cpu_count())) as pool:
            parts = pool.starmap(simulate_symbol, tasks)

    # 합치기 (빈 리스트 방지)
    trade_parts = [p[0] for p in parts if p and isinstance(p, tuple) and isinstance(p[0], pd.DataFrame) and not p[0].empty]
    stats_parts = [p[1] for p in parts if p and isinstance(p, tuple) and isinstance(p[1], pd.DataFrame) and not p[1].empty]

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    stats  = pd.concat(stats_parts,  ignore_index=True) if stats_parts  else pd.DataFrame()

    # 파일 저장
    out_trades = f"./logs/bt_tv_events_trades_{group_name}.csv"
    out_stats  = f"./logs/bt_tv_events_stats_{group_name}.csv"
    if trades.empty:
        print(f"[BT][{group_name}] trades -> {out_trades} (rows=0)")
        pd.DataFrame(columns=["event","symbol","side_used","expiry_h","ts_open","ts_close","entry","exit","result","gross","net"]).to_csv(out_trades, index=False)
    else:
        trades.to_csv(out_trades, index=False)
        print(f"[BT][{group_name}] trades -> {out_trades} (rows={len(trades)})")

    if stats.empty:
        print(f"[BT][{group_name}] stats  -> {out_stats} (rows=0)")
        pd.DataFrame(columns=["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"]).to_csv(out_stats, index=False)
    else:
        stats.to_csv(out_stats, index=False)
        print(f"[BT][{group_name}] stats  -> {out_stats} (rows={len(stats)})")

    return trades, stats


# -------------------------- Main --------------------------

def main():
    print(BANNER)
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv_enriched.csv path")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h", help="comma-separated: e.g., 0.5h,1h,2h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=0.8)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--dist-max", type=float, default=None, help="e.g., 0.02 means 2% (distance_pct <= 2)")
    ap.add_argument("--outdir", default="./logs")

    # 로컬 OHLCV 옵션
    ap.add_argument("--ohlcv-roots", default=".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv")
    ap.add_argument("--ohlcv-patterns", default="data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv")
    ap.add_argument("--assume-ohlcv-tz", default="UTC")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.signals)

    # 기본 전처리
    if "ts" not in df.columns:
        raise ValueError("signals csv must have 'ts' column")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "timeframe" not in df.columns:
        df["timeframe"] = args.timeframe
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df.rename(columns={"ticker":"symbol"}, inplace=True)
        else:
            raise ValueError("signals csv must have 'symbol' or 'ticker' column")

    # 거리 필터 (distance_pct는 % 단위; dist-max는 소수(0.02=2%))
    before = len(df)
    if args.dist_max is not None and "distance_pct" in df.columns:
        thr = args.dist_max * 100.0
        df = df[(df["distance_pct"].astype(float) <= thr)]
        print(f"[BT] distance_pct filter {args.dist_max}: {before}->{len(df)} rows")

    # 개요 출력
    n_symbols = df["symbol"].nunique()
    print(f"[BT] signals rows={len(df)}, symbols={n_symbols}, timeframe={args.timeframe}")

    # 이벤트 그룹들
    groups = ["detected", "price_in_box", "box_breakout", "line_breakout"]
    expiries_h = parse_expiries(args.expiries)

    roots = [x.strip() for x in args.ohlcv_roots.split(";") if x.strip()]
    patterns = [x.strip() for x in args.ohlcv_patterns.split(";") if x.strip()]
    assume_tz = args.assume_ohlcv_tz

    all_trades = []
    all_stats = []

    for grp in groups:
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl, args.fee,
                           expiries_h, args.procs, roots, patterns, assume_tz)
        if tr is not None and not tr.empty:
            all_trades.append(tr)
        if st is not None and not st.empty:
            all_stats.append(st)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats  = pd.concat(all_stats,  ignore_index=True) if all_stats  else pd.DataFrame()

    # 전체 Summary (이벤트+만기별 한 줄)
    if not trades.empty:
        summary = trades.groupby(["event", "expiry_h"], include_groups=False)["net"].agg(
            trades="count",
            win_rate=lambda x: (x > 0).mean(),
            avg_net="mean",
            median_net="median",
            total_net="sum",
        ).reset_index()
    else:
        summary = pd.DataFrame(columns=["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"])

    # 저장
    out_trades = os.path.join(args.outdir, "bt_tv_events_trades.csv")
    out_stats  = os.path.join(args.outdir, "bt_tv_events_stats.csv")
    trades.to_csv(out_trades, index=False)
    summary.to_csv(out_stats, index=False)

    # 출력
    if not summary.empty:
        print("\n=== Summary (by event group & expiry) ===")
        print(summary.to_string(index=False))
    else:
        print("\n=== Summary (by event group & expiry) ===")
        print("(no trades)")

    print(f"\n[BT] saved -> {out_trades} (rows={len(trades)})")
    print(f"[BT] saved -> {out_stats} (rows={len(summary)})")


if __name__ == "__main__":
    main()