#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import itertools


# -------------------------------
# Time helpers
# -------------------------------
def to_utc_series(s: pd.Series, assume_tz: Optional[str]) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce")

    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert("UTC")
    else:
        if assume_tz:
            return s.dt.tz_localize(assume_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
        else:
            return pd.to_datetime(s, utc=True, errors="coerce")


def ts_series_to_ns_utc_naive(s_utc: pd.Series) -> np.ndarray:
    if getattr(s_utc.dt, "tz", None) is None:
        s_utc = pd.to_datetime(s_utc, utc=True, errors="coerce")
    return s_utc.dt.tz_localize(None).astype("datetime64[ns]").to_numpy()


def to_ns_utc(ts: pd.Timestamp) -> np.datetime64:
    if ts.tzinfo is None:
        ts = pd.Timestamp(ts, tz="UTC")
    else:
        ts = ts.tz_convert("UTC")
    return np.datetime64(ts.tz_localize(None).to_datetime64())


# -------------------------------
# OHLCV locating & loading
# -------------------------------
def tf_variants(tf: str) -> List[str]:
    base = tf.lower()
    out = {base}
    # common aliases
    if base.endswith("m"):
        n = base[:-1]
        out |= {n, f"{n}min", f"{n}m"}
    elif base.endswith("h"):
        n = base[:-1]
        out |= {f"{n}h", f"{n}hr", f"{int(n)*60}m"}
    return list(out)


def symbol_variants(sym: str) -> List[str]:
    s = sym.upper()
    v = {s, s.replace("-", "_"), s.replace("-", ""), s.lower(), s.lower().replace("-", "_")}
    return list(v)


def expand_patterns(patterns: List[str], sym: str, tf: str) -> List[str]:
    """Replace placeholders in user patterns; also add relaxed wildcard fallbacks."""
    sym_vars = symbol_variants(sym)
    tf_vars = tf_variants(tf)
    out = []

    # exact placeholder expansion
    for p in patterns:
        for sv, tv in itertools.product(sym_vars, tf_vars):
            out.append(p.replace("{symbol}", sv).replace("{timeframe}", tv))

    # relaxed fallbacks: files containing both tokens in any order
    for sv, tv in itertools.product(sym_vars, tf_vars):
        out.append(f"**/*{sv}*{tv}*.csv")
        out.append(f"**/*{tv}*{sv}*.csv")

    # also add very relaxed by symbol only (dangerous but last resort)
    for sv in sym_vars:
        out.append(f"**/*{sv}*.csv")

    # de-dup
    seen, uniq = set(), []
    for g in out:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


def read_ohlcv_csv(path: Path, assume_tz: Optional[str]) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    alias = {
        "ts": ["ts", "time", "timestamp", "date", "datetime", "open_time", "open time"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "price"],
    }
    rename_map: Dict[str, str] = {}
    for target, names in alias.items():
        for n in names:
            if n in cols:
                rename_map[cols[n]] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    # must have
    needed = {"ts", "open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return None

    # time → UTC
    df["ts"] = to_utc_series(df["ts"], assume_tz)
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
    return df if not df.empty else None


def find_and_load_ohlcv(sym: str, tf: str, roots: List[Path], patterns: List[str], assume_tz: Optional[str]) -> Optional[pd.DataFrame]:
    globs = expand_patterns(patterns, sym, tf)
    for root in roots:
        for g in globs:
            for p in root.glob(g):
                if p.is_file():
                    df = read_ohlcv_csv(p, assume_tz)
                    if df is not None:
                        return df
    return None


# -------------------------------
# Level estimation
# -------------------------------
def estimate_level_box(
    ohlcv: pd.DataFrame,
    sig_ts_utc: pd.Timestamp,
    lookback_bars: int,
    side: str,
    q_support: float,
    q_resistance: float
) -> Optional[Tuple[float, float, float, float]]:
    ts_arr = ts_series_to_ns_utc_naive(ohlcv["ts"])
    key64 = to_ns_utc(sig_ts_utc)
    idx = int(np.searchsorted(ts_arr, key64, side="right")) - 1
    if idx < 0:
        return None

    i0 = max(0, idx - lookback_bars + 1)
    win = ohlcv.iloc[i0: idx + 1]
    if win.empty:
        return None

    closes = win["close"].astype(float)
    side = (side or "").lower()
    if side.startswith("support"):
        q_low, q_high = q_support, 0.5
    else:
        q_low, q_high = 0.5, q_resistance

    box_low = float(closes.quantile(q_low))
    box_high = float(closes.quantile(q_high))
    est_level = (box_low + box_high) / 2.0
    close_sig = float(ohlcv.iloc[idx]["close"])
    return est_level, box_low, box_high, close_sig


# -------------------------------
# Main
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Estimate TV S/R levels (adds est_level, distance_pct) from local OHLCV.")
    ap.add_argument("signals", help="signals_tv.csv")
    ap.add_argument("--timeframe", default="15m", help="Default OHLCV TF if signals have no TF column.")
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv", help="Output CSV")

    # Robustness knobs (kept for compat)
    ap.add_argument("--http-timeout", type=float, default=8.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--throttle", type=float, default=0.10)

    ap.add_argument("--lookback-bars", type=int, default=96)
    ap.add_argument("--q-support", type=float, default=0.20)
    ap.add_argument("--q-resistance", type=float, default=0.80)

    ap.add_argument("--assume-ohlcv-tz", default=None, help="e.g., Asia/Seoul")

    # NEW: where/how to find OHLCV
    ap.add_argument(
        "--ohlcv-roots",
        default=".;./ohlcv;./data;./data/ohlcv;./logs;./logs/ohlcv",
        help="Semicolon-separated root dirs to search (recursive).",
    )
    ap.add_argument(
        "--ohlcv-patterns",
        default=(
            "{symbol}-{timeframe}.csv;"
            "{symbol}_{timeframe}.csv;"
            "ohlcv/{symbol}-{timeframe}.csv;"
            "ohlcv/{symbol}_{timeframe}.csv;"
            "data/ohlcv/{symbol}-{timeframe}.csv;"
            "data/ohlcv/{symbol}_{timeframe}.csv;"
            "logs/ohlcv/{symbol}-{timeframe}.csv;"
            "logs/ohlcv/{symbol}_{timeframe}.csv"
        ),
        help="Semicolon-separated name patterns. Placeholders: {symbol}, {timeframe}. "
             "Recursive glob is used; relaxed wildcards are auto-added.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    roots = [Path(p.strip()) for p in args.ohlcv_roots.split(";") if p.strip()]
    patterns = [p.strip() for p in args.ohlcv_patterns.split(";") if p.strip()]

    # load signals
    sig_path = Path(args.signals)
    if not sig_path.exists():
        print(f"[EST] signals file not found: {sig_path}")
        return
    df = pd.read_csv(sig_path)

    # normalize columns
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc == "timestamp":
            rename_map[c] = "ts"
        elif lc in ("event", "evt"):
            rename_map[c] = "event"
        elif lc in ("side", "sr_side"):
            rename_map[c] = "side"
        elif lc in ("lvl", "level_no", "levelnum"):
            rename_map[c] = "level"
        elif lc in ("touch", "touchesmin", "min_touches"):
            rename_map[c] = "touches"
        elif lc in ("symbol", "ticker"):
            rename_map[c] = "symbol"
        elif lc in ("timeframe", "tf"):
            rename_map[c] = "timeframe"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["ts", "event", "side", "symbol"]:
        if col not in df.columns:
            print(f"[EST] missing column '{col}' in signals. Aborting.")
            return

    if "timeframe" not in df.columns or df["timeframe"].isna().all():
        df["timeframe"] = args.timeframe
    df["timeframe"] = df["timeframe"].fillna(args.timeframe).astype(str)

    # parse timestamps (TV 로그는 보통 +00:00)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # prepare outputs
    df["est_level"] = np.nan
    df["box_low"] = np.nan
    df["box_high"] = np.nan
    df["distance_pct"] = np.nan

    success = 0
    groups = list(df.groupby(["symbol", "timeframe"]).groups.keys())
    for sym, tf in groups:
        ohlcv = find_and_load_ohlcv(sym, tf, roots, patterns, args.assume_ohlcv_tz)
        if ohlcv is None or ohlcv.empty:
            print(f"[EST][{sym}] OHLCV not found for tf={tf}")
            continue

        ts_arr = ts_series_to_ns_utc_naive(ohlcv["ts"])

        rows = df[(df["symbol"] == sym) & (df["timeframe"] == tf)]
        for idx in rows.index:
            sig_ts = df.at[idx, "ts"]
            side = str(df.at[idx, "side"])

            key64 = to_ns_utc(sig_ts)
            i_bar = int(np.searchsorted(ts_arr, key64, side="right")) - 1
            if i_bar < 0:
                continue

            est = estimate_level_box(
                ohlcv=ohlcv,
                sig_ts_utc=sig_ts,
                lookback_bars=args.lookback_bars,
                side=side,
                q_support=float(args.q_support),
                q_resistance=float(args.q_resistance),
            )
            if est is None:
                continue

            est_level, box_low, box_high, close_sig = est
            df.at[idx, "est_level"] = est_level
            df.at[idx, "box_low"] = box_low
            df.at[idx, "box_high"] = box_high
            if close_sig and np.isfinite(est_level) and close_sig > 0:
                df.at[idx, "distance_pct"] = abs(close_sig - est_level) / close_sig * 100.0
            success += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[EST] done. saved -> {out}")
    print(f"[EST] success {success}/{len(df)} rows")


if __name__ == "__main__":
    main()