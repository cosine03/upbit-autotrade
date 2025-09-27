# -*- coding: utf-8 -*-
r"""
BACKTEST TV EVENTS (LOCAL_SIM v3, no external deps)

예)
python -u backtest_tv_events_mp.py .\logs\signals_tv_enriched.csv `
  --timeframe 15m `
  --expiries 0.5h,1h,2h `
  --tp 1.5 --sl 0.8 --fee 0.001 `
  --dist-max 0.02 `
  --procs 24 `
  --ohlcv-roots ".;.\data;.\data\ohlcv;.\ohlcv;.\logs;.\logs\ohlcv" `
  --ohlcv-patterns "data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv" `
  --assume-ohlcv-tz UTC `
  --outdir .\logs\bt_tv_dm0020_tp1p5_sl0p8
"""
import os
import re
import argparse
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# -------------------- pandas-safe aggregator (no include_groups) --------------------
def summarize_trades(df: pd.DataFrame, by=("event", "expiry_h")) -> pd.DataFrame:
    """
    판다스 버전과 무관하게 작동하는 집계 유틸.
    - trades: size()로 계산
    - avg_net, median_net, total_net: 표준 통계
    - win_rate: (net > 0) 비율
    """
    if df is None or len(df) == 0:
        cols = [*by, "trades", "avg_net", "median_net", "total_net", "win_rate"]
        return pd.DataFrame(columns=cols)

    g = df.groupby(list(by), dropna=False)
    trades_ct = g.size().reset_index(name="trades")
    stats = (
        g["net"]
        .agg(["mean", "median", "sum"])
        .reset_index()
        .rename(columns={"mean": "avg_net", "median": "median_net", "sum": "total_net"})
    )
    win_rate = g["net"].apply(lambda s: (s > 0).mean()).reset_index(name="win_rate")

    out = trades_ct.merge(stats, on=list(by), how="left").merge(win_rate, on=list(by), how="left")
    return out.fillna(
        {"trades": 0, "avg_net": 0.0, "median_net": 0.0, "total_net": 0.0, "win_rate": 0.0}
    )

# -------------------- 로컬 OHLCV 로딩 --------------------
def _ensure_ts_utc(df: pd.DataFrame, assume_tz: str = "UTC") -> pd.DataFrame:
    out = df.copy()
    if "ts" in out.columns:
        ts = pd.to_datetime(out["ts"], errors="coerce", utc=False)
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(assume_tz).dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
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

def _load_csv(path: str, assume_tz: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        lower = {c: c.lower() for c in df.columns}
        df.rename(columns=lower, inplace=True)
        need = {"open", "high", "low", "close"}
        if not need.issubset(df.columns):
            return None
        df = _ensure_ts_utc(df, assume_tz=assume_tz)
        keep = ["ts", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
        df = df[keep].dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
        return df
    except Exception:
        return None

def get_ohlcv_csv(symbol: str, timeframe: str, roots: List[str], patterns: List[str], assume_tz: str) -> Optional[pd.DataFrame]:
    for root in roots:
        rroot = root.strip()
        for pat in patterns:
            rel = pat.strip().format(symbol=symbol, timeframe=timeframe)
            if not rel.lower().endswith(".csv"):
                rel += ".csv"
            path = os.path.normpath(os.path.join(rroot, rel) if rroot else rel)
            df = _load_csv(path, assume_tz=assume_tz)
            if df is not None and not df.empty:
                print(f"[BT][LOCAL] {symbol} {timeframe} -> {path} (rows={len(df)})")
                return df
    return None

# -------------------- 유틸 --------------------
def to_utc_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def parse_side(row: pd.Series) -> Optional[str]:
    side = str(row.get("side", "") or "").strip().lower()
    if side in ("support", "resistance"):
        return side
    msg = str(row.get("message", "") or "")
    if re.search(r"resistance", msg, re.I):
        return "resistance"
    if re.search(r"support", msg, re.I):
        return "support"
    return None

# -------------------- 시뮬 엔진 (로컬 간단 버전) --------------------
def _simulate_one_trade(ohlcv: pd.DataFrame, ts_sig: pd.Timestamp, tp_pct: float, sl_pct: float, fee: float, expiry_h: float, side: str) -> Optional[float]:
    ts = pd.to_datetime(ohlcv["ts"], utc=True, errors="coerce")
    ts_idx = pd.DatetimeIndex(ts)
    idx = int(ts_idx.searchsorted(ts_sig, side="right") - 1)
    if idx < 0 or idx >= len(ohlcv):
        return None

    entry = float(ohlcv["close"].iloc[idx])
    end_ts = ts_sig + pd.Timedelta(hours=expiry_h)

    look = ohlcv.iloc[idx + 1:].copy()
    if look.empty:
        return None
    look = look[look["ts"] <= end_ts]
    if look.empty:
        return None

    tp_price = entry * (1 + tp_pct / 100.0)
    sl_price = entry * (1 - sl_pct / 100.0)

    hit_tp_idx = None
    hit_sl_idx = None
    highs = look["high"].to_numpy(dtype=float)
    lows  = look["low"].to_numpy(dtype=float)

    for i in range(len(look)):
        if highs[i] >= tp_price:
            hit_tp_idx = i
            break
        if lows[i] <= sl_price:
            hit_sl_idx = i
            break

    round_fee = 2 * fee

    if hit_tp_idx is None and hit_sl_idx is None:
        exit_px = float(look["close"].iloc[-1])
        return float((exit_px - entry) / entry - round_fee)

    if hit_tp_idx is not None and (hit_sl_idx is None or hit_tp_idx <= hit_sl_idx):
        return float((tp_price - entry) / entry - round_fee)
    else:
        return float((sl_price - entry) / entry - round_fee)

def simulate_symbol(symbol: str,
                    df_sym: pd.DataFrame,
                    timeframe: str,
                    tp: float, sl: float, fee: float,
                    expiries_h: List[float],
                    roots: List[str], patterns: List[str], assume_tz: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ohlcv = get_ohlcv_csv(symbol, timeframe, roots, patterns, assume_tz)
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for _, r in df_sym.iterrows():
        ts_sig = to_utc_ts(r["ts"])
        side = parse_side(r) or "support"
        event = str(r.get("event", "detected"))
        for eh in expiries_h:
            net = _simulate_one_trade(ohlcv, ts_sig, tp, sl, fee, eh, side)
            if net is None:
                continue
            rows.append({
                "symbol": symbol,
                "event": event,
                "expiry_h": float(eh),
                "net": float(net),
            })

    trades_df = pd.DataFrame(rows)
    if trades_df.empty:
        return trades_df, pd.DataFrame(columns=["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"])

    summary_df = summarize_trades(trades_df, by=("event", "expiry_h"))
    return trades_df, summary_df

# -------------------- 그룹 실행 --------------------
def run_group(df: pd.DataFrame, group_name: str, timeframe: str,
              tp: float, sl: float, fee: float, expiries_h: List[float],
              procs: int, roots: List[str], patterns: List[str], assume_tz: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["event_group"] == group_name].copy()
    if sub.empty:
        print(f"[BT][{group_name}] no tasks.")
        return pd.DataFrame(), pd.DataFrame()

    symbols = sorted(sub["symbol"].unique().tolist())
    tasks = []
    for sym in symbols:
        tasks.append((
            sym,
            sub[sub["symbol"] == sym].copy(),
            timeframe, tp, sl, fee,
            expiries_h,
            roots, patterns, assume_tz
        ))

    if procs <= 1:
        parts = [simulate_symbol(*t) for t in tasks]
    else:
        with Pool(processes=procs) as pool:
            parts = pool.starmap(simulate_symbol, tasks)

# 수정 (무트레이드 가드)
tr_list = [p[0] for p in parts if p and isinstance(p[0], pd.DataFrame) and not p[0].empty]
st_list = [p[1] for p in parts if p and isinstance(p[1], pd.DataFrame) and not p[1].empty]

if not tr_list:
    empty_tr = pd.DataFrame(columns=["symbol","event","ts","entry","exit","pnl","net","expiry_h"])
    empty_st = pd.DataFrame(columns=["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"])
    return empty_tr, empty_st

trades = pd.concat(tr_list, ignore_index=True)
stats  = pd.concat(st_list, ignore_index=True) if st_list else pd.DataFrame(
    columns=["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"]
)
return trades, stats

# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv_enriched.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="0.5h,1h,2h")
    ap.add_argument("--tp", type=float, default=1.5, help="take-profit in percent (1.5 = 1.5%)")
    ap.add_argument("--sl", type=float, default=0.8, help="stop-loss in percent (0.8 = 0.8%)")
    ap.add_argument("--fee", type=float, default=0.001, help="per-side fee (0.001 = 0.1%)")
    ap.add_argument("--procs", type=int, default=max(1, cpu_count() // 2))
    ap.add_argument("--dist-max", type=float, default=0.02, help="max distance ratio (0.02 => 2%)")
    ap.add_argument("--outdir", default="./logs/bt_tv_out")
    # 로컬 CSV
    ap.add_argument("--ohlcv-roots", default=".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv")
    ap.add_argument("--ohlcv-patterns", default="data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv")
    ap.add_argument("--assume-ohlcv-tz", default="UTC")
    args = ap.parse_args()

    print("[BT] BACKTEST TV EVENTS (LOCAL_SIM v3, no external deps)")

    df = pd.read_csv(args.signals)
    if "ts" not in df.columns:
        raise ValueError("signals 파일에 'ts'가 필요합니다.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df.rename(columns={"ticker": "symbol"}, inplace=True)
        else:
            raise ValueError("'symbol' 컬럼이 필요합니다.")
    if "event" not in df.columns:
        df["event"] = "detected"

    # 거리 필터: distance_pct 단위 자동 감지 (ratio vs percent)
    if "distance_pct" in df.columns and pd.api.types.is_numeric_dtype(df["distance_pct"]):
        before = len(df)
        dp = pd.to_numeric(df["distance_pct"], errors="coerce")
        mx = float(dp.max()) if len(dp) else float("nan")
        if pd.isna(mx):
            thr = args.dist_max
            scale = "unknown->ratio"
        elif mx <= 1.0:        # 0.002 = 0.2% 같은 비율 스케일
            thr = args.dist_max
            scale = "ratio"
        else:                  # 2.0 = 2% 같은 퍼센트 스케일
            thr = args.dist_max * 100.0
            scale = "percent"
        df = df[dp <= thr].copy()
        print(f"[BT] distance_pct filter ({scale}) thr={thr:.6g}: {before}->{len(df)} rows")

    if df.empty:
        print("[BT] signals empty after filter.")
        return

    def _group(e: str) -> str:
        e = (e or "").lower()
        if "price_in_box" in e: return "price_in_box"
        if "box_breakout" in e: return "box_breakout"
        if "line_breakout" in e: return "line_breakout"
        return "detected"

    df["event_group"] = df["event"].astype(str).map(_group)

    expiries_h = []
    for token in str(args.expiries).split(","):
        s = token.strip().lower()
        if not s:
            continue
        if s.endswith("h"):
            expiries_h.append(float(s[:-1]))
        elif s.endswith("m"):
            expiries_h.append(float(s[:-1]) / 60.0)
        else:
            expiries_h.append(float(s))
    expiries_h = sorted(set(expiries_h))

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    roots = [x.strip() for x in args.ohlcv_roots.split(";") if x.strip()]
    patterns = [x.strip() for x in args.ohlcv_patterns.split(";") if x.strip()]
    assume_tz = args.assume_ohlcv_tz

    print(f"[BT] signals rows={len(df)}, symbols={df['symbol'].nunique()}, timeframe={args.timeframe}")

    all_trades = []
    all_stats  = []
    for grp in ["detected", "price_in_box", "box_breakout", "line_breakout"]:
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl, args.fee,
                           expiries_h, args.procs, roots, patterns, assume_tz)
        if not tr.empty:
            tr["group"] = grp
            all_trades.append(tr)
        if not st.empty:
            st["group"] = grp
            all_stats.append(st)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats  = pd.concat(all_stats,  ignore_index=True) if all_stats  else pd.DataFrame()

    if not trades.empty:
        trades_path = os.path.join(outdir, "bt_tv_events_trades.csv")
        trades.to_csv(trades_path, index=False)
        print(f"[BT] trades saved -> {trades_path}")
    if not stats.empty:
        stats_raw_path = os.path.join(outdir, "bt_tv_events_stats_raw.csv")
        stats.to_csv(stats_raw_path, index=False)
        print(f"[BT] stats(raw) saved -> {stats_raw_path}")

    if not trades.empty:
        gsum = summarize_trades(trades, by=("event", "expiry_h"))
        gsum = gsum[["event", "expiry_h", "trades", "win_rate", "avg_net", "median_net", "total_net"]]
        pd.options.display.float_format = lambda v: f"{v:,.6f}"
        print("\n[BT] SUMMARY (by event, expiry_h)")
        print(gsum.to_string(index=False))
        summary_path = os.path.join(outdir, "bt_tv_events_stats_summary.csv")
        gsum.to_csv(summary_path, index=False)
        print(f"[BT] summary saved -> {summary_path}")

    print(f"\n[BT] done -> {outdir}")

if __name__ == "__main__":
    main()
