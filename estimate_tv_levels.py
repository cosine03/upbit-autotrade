#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np


# -------------------------------
# Helpers: time & array utilities
# -------------------------------

def to_utc_series(s: pd.Series, assume_tz: Optional[str]) -> pd.Series:
    """
    Convert a timestamp series to UTC tz-aware series.
    - If assume_tz is provided and s is tz-naive -> localize to assume_tz then convert to UTC.
    - If s is already tz-aware -> convert to UTC.
    - Else (naive & no assume_tz) -> parse with utc=True (assume input is UTC).
    """
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce")

    if getattr(s.dt, "tz", None) is not None:
        # tz-aware -> convert to UTC
        return s.dt.tz_convert("UTC")
    else:
        if assume_tz:
            return s.dt.tz_localize(assume_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
        else:
            # treat as already UTC
            return pd.to_datetime(s, utc=True, errors="coerce")


def ts_series_to_ns_utc_naive(s_utc: pd.Series) -> np.ndarray:
    """
    Convert a UTC tz-aware timestamp series to numpy datetime64[ns] (tz-naive) for fast searchsorted.
    """
    if getattr(s_utc.dt, "tz", None) is None:
        # If somehow naive here, ensure utc=True then drop tz
        s_utc = pd.to_datetime(s_utc, utc=True, errors="coerce")
    # drop tz -> ns
    return s_utc.dt.tz_localize(None).astype("datetime64[ns]").to_numpy()


def to_ns_utc(ts: pd.Timestamp) -> np.datetime64:
    """Ensure a single timestamp is UTC tz-aware then return datetime64[ns] naive."""
    if ts.tzinfo is None:
        ts = pd.Timestamp(ts, tz="UTC")
    else:
        ts = ts.tz_convert("UTC")
    return np.datetime64(ts.tz_localize(None).to_datetime64())


# -------------------------------
# OHLCV loading
# -------------------------------

def candidate_paths(symbol: str, timeframe: str) -> List[Path]:
    """
    Return plausible OHLCV paths to try.
    """
    bases = [Path("./ohlcv"), Path("./data/ohlcv"), Path("./logs/ohlcv"), Path("./data")]
    names = [
        f"{symbol}-{timeframe}.csv",
        f"{symbol}_{timeframe}.csv",
        f"{symbol}-{timeframe}.CSV",
        f"{symbol}_{timeframe}.CSV",
    ]
    paths = []
    for b in bases:
        for n in names:
            p = b / n
            paths.append(p)
    return paths


def load_ohlcv(symbol: str, timeframe: str, assume_tz: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load OHLCV CSV for symbol/timeframe from common locations.
    Expected columns: ts/open/high/low/close (ts can be string or datetime)
    Returns None if not found or invalid.
    """
    for p in candidate_paths(symbol, timeframe):
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue

            cols_lower = {c.lower(): c for c in df.columns}
            req = ["ts", "open", "high", "low", "close"]
            if not all(c in (k.lower() for k in df.columns) for c in req):
                # Try to normalize column names
                rename_map: Dict[str, str] = {}
                for want in req:
                    if want in cols_lower:
                        rename_map[cols_lower[want]] = want
                if rename_map:
                    df = df.rename(columns=rename_map)

            # Validate again
            if not set(["ts", "open", "high", "low", "close"]).issubset(df.columns):
                continue

            # Time parse & UTC normalize
            df["ts"] = to_utc_series(df["ts"], assume_tz)
            # Sort & drop na
            df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
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
    """
    Given OHLCV and a signal timestamp (UTC tz-aware), take the previous `lookback_bars`
    and estimate a support/resistance box via quantiles.

    Returns tuple: (est_level, box_low, box_high, close_at_signal)
    or None if insufficient data.
    """
    ts_arr = ts_series_to_ns_utc_naive(ohlcv["ts"])
    key64 = to_ns_utc(sig_ts_utc)
    # index of bar <= sig_ts (right-1)
    idx = int(np.searchsorted(ts_arr, key64, side="right")) - 1
    if idx < 0:
        return None

    i0 = max(0, idx - lookback_bars + 1)
    window = ohlcv.iloc[i0 : idx + 1]
    if window.empty:
        return None

    # price proxies: use 'close' distribution to infer level area
    closes = window["close"].astype(float)

    if side.lower().startswith("support"):
        q_low = q_support
        q_high = 0.5  # median as top of support box
    else:
        # resistance
        q_low = 0.5  # median as bottom of resistance box
        q_high = q_resistance

    box_low = float(closes.quantile(q_low))
    box_high = float(closes.quantile(q_high))

    # est_level: midpoint of the box
    est_level = (box_low + box_high) / 2.0

    close_at_signal = float(ohlcv.iloc[idx]["close"])
    return est_level, box_low, box_high, close_at_signal


# -------------------------------
# Main
# -------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Estimate TV S/R levels for signals (adds est_level, distance_pct).")
    ap.add_argument("signals", help="signals_tv.csv")
    ap.add_argument("--timeframe", default="15m", help="OHLCV timeframe to use (default: 15m)")
    ap.add_argument("--out", default="./logs/signals_tv_enriched.csv", help="Output CSV path")

    # Robustness / performance knobs (kept for interface compatibility)
    ap.add_argument("--http-timeout", type=float, default=8.0, help="(unused placeholder) http timeout seconds")
    ap.add_argument("--retries", type=int, default=4, help="(unused placeholder) retry count")
    ap.add_argument("--throttle", type=float, default=0.10, help="(unused placeholder) throttle seconds")

    ap.add_argument("--lookback-bars", type=int, default=96, help="Bars to look back before each signal (default: 96)")
    ap.add_argument("--q-support", type=float, default=0.20, help="Support box lower quantile (default: 0.20)")
    ap.add_argument("--q-resistance", type=float, default=0.80, help="Resistance box upper quantile (default: 0.80)")

    ap.add_argument("--assume-ohlcv-tz", default=None,
                    help="Assume OHLCV timestamps are in this timezone (e.g., 'Asia/Seoul'); convert to UTC internally.")

    return ap.parse_args()


def main():
    args = parse_args()

    # Load signals
    sig_path = Path(args.signals)
    if not sig_path.exists():
        print(f"[EST] signals file not found: {sig_path}")
        return

    df = pd.read_csv(sig_path)
    # Normalize column names
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

    required = ["ts", "event", "side", "symbol"]
    for col in required:
        if col not in df.columns:
            print(f"[EST] missing column '{col}' in signals. Aborting.")
            return

    # Ensure timeframe column
    if "timeframe" not in df.columns or df["timeframe"].isna().all():
        df["timeframe"] = args.timeframe
    df["timeframe"] = df["timeframe"].fillna(args.timeframe).astype(str)

    # Parse signals ts as UTC (strings from TV have '+00:00')
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # Prepare output columns
    df["est_level"] = np.nan
    df["box_low"] = np.nan
    df["box_high"] = np.nan
    df["distance_pct"] = np.nan

    # Work per symbol & timeframe (load ohlcv once per pair)
    success = 0
    total = 0

    for (sym, tf), rows in df.groupby(["symbol", "timeframe"]):
        ohlcv = load_ohlcv(sym, tf, assume_tz=args.assume_ohlcv_tz)
        if ohlcv is None or ohlcv.empty:
            print(f"[EST][{sym}] OHLCV not found or empty for tf={tf}")
            continue

        # For quick search
        ts_arr = ts_series_to_ns_utc_naive(ohlcv["ts"])

        for idx in rows.index:
            total += 1
            sig_ts = df.at[idx, "ts"]  # tz-aware UTC
            side = str(df.at[idx, "side"]).strip().lower()  # 'support' or 'resistance'

            # Find bar <= sig_ts
            key64 = to_ns_utc(sig_ts)
            i_bar = int(np.searchsorted(ts_arr, key64, side="right")) - 1
            if i_bar < 0:
                continue

            # Estimate box from lookback
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

            # Populate
            df.at[idx, "est_level"] = est_level
            df.at[idx, "box_low"] = box_low
            df.at[idx, "box_high"] = box_high
            # distance relative to close
            if close_sig and np.isfinite(est_level) and close_sig > 0:
                df.at[idx, "distance_pct"] = abs(close_sig - est_level) / close_sig * 100.0
            success += 1

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[EST] done. saved -> {out_path}")
    print(f"[EST] success {success}/{len(df)} rows")


if __name__ == "__main__":
    main()