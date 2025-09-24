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
import time
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from utils.agg import summarize_trades  # [PATCH] pandas-safe groupby aggregator

# -------------------- 로컬 OHLCV 로딩 --------------------

def _ensure_ts_utc(df: pd.DataFrame, assume_tz: str = "UTC") -> pd.DataFrame:
    out = df.copy()
    if "ts" in out.columns:
        ts = pd.to_datetime(out["ts"], errors="coerce")
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
        return df[keep].dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
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
# 가정:
# - 입력 CSV(signals_tv_enriched.csv)에 ts, symbol, event, side, distance_pct, est_level, sig_price 가 있음
# - 각 신호(ts) 직후부터 expiry_h 시간 내 TP/SL 체크 (간단화: 고/저가 스캔), 수수료 반영

def _simulate_one_trade(ohlcv: pd.DataFrame, ts_sig: pd.Timestamp, tp_pct: float, sl_pct: float, fee: float, expiry_h: float, side: str) -> Optional[float]:
    # 시그널 봉 인덱스
    ts = pd.to_datetime(ohlcv["ts"], utc=True, errors="coerce")
    idx = int(np.searchsorted(ts.to_numpy(), np.datetime64(ts_sig), side="right") - 1)
    if idx < 0 or idx >= len(ohlcv):
        return None

    entry = float(ohlcv["close"].iloc[idx])
    # 유효 구간
    end_ts = ts_sig + pd.Timedelta(hours=expiry_h)
    # 다음 봉부터 스캔
    look = ohlcv.iloc[idx+1:].copy()
    if look.empty:
        return None
    look = look[look["ts"] <= end_ts]
    if look.empty:
        return None

    # 목표/손절 가격
    if side == "support":
        tp_price = entry * (1 + tp_pct/100.0)
        sl_price = entry * (1 - sl_pct/100.0)
    else:  # resistance 기준 역방향이 아니라 동일 방향(돌파 추종)이라고 가정
        tp_price = entry * (1 + tp_pct/100.0)
        sl_price = entry * (1 - sl_pct/100.0)

    hit_tp_idx = None
    hit_sl_idx = None

    # 봉 내 고저를 이용해 최초 도달 시점 판정
    highs = look["high"].to_numpy(dtype=float)
    lows  = look["low"].to_numpy(dtype=float)

    for i in range(len(look)):
        if highs[i] >= tp_price:
            hit_tp_idx = i
            break
        if lows[i] <= sl_price:
            hit_sl_idx = i
            break

    # 수익률(수수료 왕복 2회 가정: 진입/청산)
    round_fee = 2 * fee

    if hit_tp_idx is None and hit_sl_idx is None:
        # 타임아웃: 마지막 종가 청산
        exit_px = float(look["close"].iloc[-1])
        ret = (exit_px - entry) / entry - round_fee
        return float(ret)

    if hit_tp_idx is not None and (hit_sl_idx is None or hit_tp_idx <= hit_sl_idx):
        # TP 선도달
        ret = (tp_price - entry) / entry - round_fee
        return float(ret)
    else:
        # SL 선도달
        ret = (sl_price - entry) / entry - round_fee
        return float(ret)


def simulate_symbol(symbol: str,
                    df_sym: pd.DataFrame,
                    timeframe: str,
                    tp: float, sl: float, fee: float,
                    expiries_h: List[float],
                    roots: List[str], patterns: List[str], assume_tz: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ohlcv = get_ohlcv_csv(symbol, timeframe, roots, patterns, assume_tz)
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(), pd.DataFrame()

    trades = []
    for _, r in df_sym.iterrows():
        ts_sig = to_utc_ts(r["ts"])
        side = parse_side(r) or "support"
        event = str(r.get("event", "detected"))
        for eh in expiries_h:
            net = _simulate_one_trade(ohlcv, ts_sig, tp, sl, fee, eh, side)
            if net is None:
                continue
            trades.append({
                "symbol": symbol,
                "event": event,
                "expiry_h": float(eh),
                "net": float(net),
            })
    trades_df = pd.DataFrame(trades)

    # --- 요약 통계 (판다스 하위버전 호환: include_groups 미사용) ---
# [PATCH] pandas-compat summary (no include_groups)
try:
    # 신판다스에서만 되는 경로 (가능하면 사용)
    summary = trades.groupby(["event", "expiry_h"], include_groups=False)["net"].agg(
        trades=("net", "count"),
        win_rate=(lambda s: (s > 0).mean()),
        avg_net=("net", "mean"),
        median_net=("net", "median"),
        total_net=("net", "sum"),
    ).reset_index()
except TypeError:
    # 구판다스/혼합환경 안전 경로
    summary = summarize_trades(trades, by=("event", "expiry_h"))

# -------------------- 메인 --------------------

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

    # 합치기 (빈 경우 안전 처리)
    trades = pd.concat([p[0] for p in parts if p and isinstance(p[0], pd.DataFrame) and not p[0].empty],
                       ignore_index=True) if parts else pd.DataFrame()
    stats  = pd.concat([p[1] for p in parts if p and isinstance(p[1], pd.DataFrame) and not p[1].empty],
                       ignore_index=True) if parts else pd.DataFrame()

    return trades, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv_enriched.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="0.5h,1h,2h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=0.8)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--procs", type=int, default=max(1, cpu_count() // 2))
    ap.add_argument("--dist-max", type=float, default=0.02)
    ap.add_argument("--outdir", default="./logs/bt_tv_out")
    # 로컬 CSV
    ap.add_argument("--ohlcv-roots", default=".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv")
    ap.add_argument("--ohlcv-patterns", default="data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv")
    ap.add_argument("--assume-ohlcv-tz", default="UTC")
    args = ap.parse_args()

    print("[BT] BACKTEST TV EVENTS (LOCAL_SIM v3, no external deps)")

    # 입력 로드
    df = pd.read_csv(args.signals)
    # 표준화
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

    # 거리 필터
    if "distance_pct" in df.columns and pd.api.types.is_numeric_dtype(df["distance_pct"]):
        before = len(df)
        df = df[df["distance_pct"] <= (args.dist_max * 100.0)].copy()
        print(f"[BT] distance_pct filter {args.dist_max}: {before}->{len(df)} rows")

    if df.empty:
        print("[BT] signals empty after filter.")
        return

    # 그룹 매핑
    def _group(e: str) -> str:
        e = (e or "").lower()
        if "price_in_box" in e: return "price_in_box"
        if "box_breakout" in e: return "box_breakout"
        if "line_breakout" in e: return "line_breakout"
        return "detected"

    df["event_group"] = df["event"].astype(str).map(_group)

    # 만료 시간 해석
    expiries_h = []
    for token in str(args.expiries).split(","):
        s = token.strip().lower()
        if s.endswith("h"):
            expiries_h.append(float(s[:-1]))
        elif s.endswith("m"):
            expiries_h.append(float(s[:-1]) / 60.0)
        else:
            expiries_h.append(float(s))
    expiries_h = sorted(set(expiries_h))

    # 출력 디렉토리
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # roots/patterns
    roots = [x.strip() for x in args.ohlcv_roots.split(";") if x.strip()]
    patterns = [x.strip() for x in args.ohlcv_patterns.split(";") if x.strip()]
    assume_tz = args.assume_ohlcv_tz

    print(f"[BT] signals rows={len(df)}, symbols={df['symbol'].nunique()}, timeframe={args.timeframe}")

    # 그룹별 실행
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

    # 저장
    if not trades.empty:
        trades.to_csv(os.path.join(outdir, "bt_tv_events_trades.csv"), index=False)
    if not stats.empty:
        stats.to_csv(os.path.join(outdir, "bt_tv_events_stats_raw.csv"), index=False)

    # 요약(그룹/만료)
    if not stats.empty:
        # 동일 포맷으로 출력
        stats = stats[["group", "event", "expiry_h", "trades", "win_rate", "avg_net", "median_net", "total_net"]]
        # 화면 요약
    # [PATCH] pandas-compat summary (no include_groups)
    try:
    # 신판다스에서만 되는 경로 (가능하면 사용)
        summary = trades.groupby(["event", "expiry_h"], include_groups=False)["net"].agg(
            trades=("net", "count"),
            win_rate=(lambda s: (s > 0).mean()),
            avg_net=("net", "mean"),
            median_net=("net", "median"),
            total_net=("net", "sum"),
        ).reset_index()
    except TypeError:
        # 구판다스/혼합환경 안전 경로
        summary = summarize_trades(trades, by=("event", "expiry_h"))

        # 보기 좋게 소수점 조정
        pd.options.display.float_format = lambda v: f"{v:,.6f}"
        print(gsum.to_string(index=False))

        # 별도로 저장
        gsum.to_csv(os.path.join(outdir, "bt_tv_events_stats_summary.csv"), index=False)

    print(f"\n[BT] saved -> {outdir}")