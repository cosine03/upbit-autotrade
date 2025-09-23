# -*- coding: utf-8 -*-
"""
TV breakout(close-based) backtest, multiprocessing + throttled OHLCV fetch.

- Entry: TV 신호봉 다음 봉의 '종가가 레벨 위에서 마감'(롱 기준)일 때 진입 (Breakout = close above level)
- Exit: 24h 만기 또는 TP/SL 없음(폴 규칙 단순 비교용) — 필요 시 확장 가능
- 데이터: Upbit OHLCV (sr_engine.data.get_ohlcv)
- 멀티프로세싱: 프로세스 수 (--procs), 동시 OHLCV 호출 수 (--max-fetch-concurrency) 분리 제어
- 타임존: 전부 UTC tz-aware로 통일. searchsorted 비교는 tz-naive numpy(datetime64[ns])로 변환한 뒤 수행

Run:
  python backtest_tv_breakout_close_mp.py .\logs\signals_tv.csv --timeframe 15m --procs 6 --max-fetch-concurrency 2

Tips:
- 네트워크가 느리면 --procs 4~6, --max-fetch-concurrency 1~2 로 시작 권장
- 문제가 있는 심볼은 자동 skip
"""

import os
import sys
import time
import math
import argparse
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
from multiprocessing import get_context

# 우리 프로젝트 유틸
from sr_engine.data import get_ohlcv  # Upbit OHLCV
pd.set_option("future.no_silent_downcasting", True)

# -------- Time helpers --------
def to_utc_ts(x) -> pd.Timestamp:
    """Any → tz-aware UTC Timestamp"""
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts

def utc_index_to_numpy64(ts_index: pd.Index) -> np.ndarray:
    """UTC tz-aware DatetimeIndex → numpy datetime64[ns] (tz-naive)"""
    # pandas가 내부적으로 UTC로 보정한 뒤 naive ns 배열로 뽑아옴
    return ts_index.tz_convert("UTC").astype("datetime64[ns]").to_numpy()

def search_right(ts64: np.ndarray, when_utc: pd.Timestamp) -> int:
    """searchsorted right on tz-naive numpy64 timeline"""
    return int(np.searchsorted(ts64, np.datetime64(when_utc.to_pydatetime()), side="right"))

def search_left(ts64: np.ndarray, when_utc: pd.Timestamp) -> int:
    return int(np.searchsorted(ts64, np.datetime64(when_utc.to_pydatetime()), side="left"))

# -------- Robust OHLCV fetch (throttled in worker via global semaphore) --------
_SEM = None
HTTP_RETRIES = 3
HTTP_BACKOFF = 0.8  # seconds

def set_throttle(sem, retries: int, backoff: float):
    global _SEM, HTTP_RETRIES, HTTP_BACKOFF
    _SEM = sem
    HTTP_RETRIES = max(1, int(retries))
    HTTP_BACKOFF = max(0.05, float(backoff))

def get_ohlcv_safe(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Throttle + retry wrapper around sr_engine.data.get_ohlcv"""
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            if _SEM is not None:
                with _SEM:
                    df = get_ohlcv(symbol, timeframe)
            else:
                df = get_ohlcv(symbol, timeframe)
            if df is None or df.empty:
                raise RuntimeError("empty OHLCV")
            # 표준 컬럼 보정
            if "ts" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    idx = df.index
                else:
                    # sr_engine.data가 보장해주지만, 혹시몰라서
                    idx = pd.to_datetime(df["time"], utc=True, errors="coerce") if "time" in df.columns else pd.to_datetime(df.iloc[:,0], utc=True, errors="coerce")
                idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
                df = df.copy()
                df["ts"] = idx
                df = df.reset_index(drop=True)
            else:
                df = df.copy()
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
            # 정렬
            df = df.sort_values("ts").reset_index(drop=True)
            return df
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt >= HTTP_RETRIES:
                print(f"[{symbol}] OHLCV failed after {attempt} tries: {e}")
                return None
            sleep_s = HTTP_BACKOFF * (2 ** (attempt - 1))
            print(f"[{symbol}] OHLCV load error (try {attempt}/{HTTP_RETRIES}): {repr(e)} -> retry in {sleep_s:.2f}s")
            time.sleep(sleep_s)

# -------- Breakout entry (close-based) --------
def breakout_close_entries(df: pd.DataFrame, signals: pd.DataFrame, touches_required: int = 2) -> list[dict]:
    """
    TV 로그(row)에 들어있는 level(center)와 side(=resistance/support)를 이용.
    다음 봉들의 종가가 '레벨 위에서 마감'(롱 기준: resistance를 돌파) 했을 때 엔트리 생성.
    여기서는 비교 실험을 위해 touches_required(2 또는 3)만 필터로 사용.
    """
    out = []
    if df is None or df.empty or signals.empty:
        return out

    # 시간축
    ts64 = utc_index_to_numpy64(pd.DatetimeIndex(df["ts"]))
    close = df["close"].to_numpy(dtype=float)

    for _, s in signals.iterrows():
        try:
            # 신호 정보
            sig_ts = to_utc_ts(s["ts"])
            # message/extra에서 center(=level) 복원: 이미 pre-parsed 되어 있다고 가정, 없으면 skip
            # (이 스크립트에서는 signals에 'center' 컬럼이 있다고 가정. 없으면 simple parse 추가)
            center = float(s.get("center", np.nan))
            touches = int(s.get("touches", 0))
            side = str(s.get("side", "")).lower()

            if not np.isfinite(center):
                # message에서 center 파싱 시도
                msg = str(s.get("message", ""))
                # 예: "... center=6310458.3333 | band=8375.0"
                import re
                m = re.search(r"center=([0-9]+(?:\.[0-9]+)?)", msg)
                if m:
                    center = float(m.group(1))
                else:
                    continue

            if touches < touches_required:
                continue

            # 엔트리 판단: 신호 이후의 '다음 봉'부터 검사
            i_start = search_right(ts64, sig_ts)
            if i_start >= len(df):
                continue

            # "종가가 레벨 위에서 마감" (롱, resistance 돌파 기준으로 단순화)
            # 폴 정의에 충실하려면 side가 'resistance'일 때만 본다(지원/저항 혼재 시 혼선 방지)
            if side != "resistance":
                continue

            # i_start부터 순차로 첫 breakout close 찾기
            ent_idx = None
            for i in range(i_start, len(df)):
                if close[i] > center:
                    ent_idx = i
                    break
            if ent_idx is None:
                continue

            out.append({
                "entry_ts": df["ts"].iloc[ent_idx],
                "entry_px": close[ent_idx],
                "symbol": s["symbol"],
                "level": center,
                "touches": touches,
                "sig_ts": sig_ts,
                "side": side,
            })
        except Exception as e:
            print(f"[{s.get('symbol','?')}] breakout parse error: {e}")
            continue

    return out

# -------- Exit rule: 24h expiry close --------
def exit_by_expiry(df: pd.DataFrame, entry_ts: pd.Timestamp, hours: int = 24) -> tuple[pd.Timestamp, float] | None:
    if df is None or df.empty:
        return None
    ts64 = utc_index_to_numpy64(pd.DatetimeIndex(df["ts"]))
    exit_ts = to_utc_ts(entry_ts) + timedelta(hours=hours)
    idx_exit = search_left(ts64, exit_ts)
    if idx_exit >= len(df):
        idx_exit = len(df) - 1
    return df["ts"].iloc[idx_exit], float(df["close"].iloc[idx_exit])

# -------- Worker --------
def worker(args):
    """
    args: (symbol, timeframe, sig_rows_df, touches_required)
    """
    symbol, timeframe, sig_rows, touches_required = args
    try:
        df = get_ohlcv_safe(symbol, timeframe)
        if df is None:
            return pd.DataFrame(), pd.DataFrame()

        # 엔트리 추출
        entries = breakout_close_entries(df, sig_rows, touches_required=touches_required)
        if not entries:
            return pd.DataFrame(), pd.DataFrame()

        # 트레이드 생성 (24h 만기, 수수료 0.1% 왕복)
        FEE = 0.001
        trades = []
        for e in entries:
            ex = exit_by_expiry(df, e["entry_ts"], hours=24)
            if not ex:
                continue
            ex_ts, ex_px = ex
            entry_px = float(e["entry_px"])
            net = (ex_px - entry_px) / entry_px - FEE
            trades.append({
                "symbol": e["symbol"],
                "entry_ts": e["entry_ts"],
                "entry_px": entry_px,
                "exit_ts": ex_ts,
                "exit_px": ex_px,
                "net": net,
                "touches": e["touches"],
                "level": e["level"],
            })

        if not trades:
            return pd.DataFrame(), pd.DataFrame()

        tdf = pd.DataFrame(trades)
        # 통계
        wins = (tdf["net"] > 0).mean() if not tdf.empty else 0.0
        stats = pd.DataFrame([{
            "symbol": symbol,
            "trades": len(tdf),
            "win_rate": float(wins),
            "avg_net": float(tdf["net"].mean()),
            "median_net": float(tdf["net"].median()),
            "total_net": float(tdf["net"].sum()),
            "touches_required": touches_required,
        }])
        return tdf, stats

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[{symbol}] worker error: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

# -------- Main --------
def load_tv_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 표준 컬럼 보정
    if "ts" not in df.columns:
        raise ValueError("signals_tv.csv must contain 'ts' column")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    # 필터링: breakout 관련만(저항 레벨 터치/박스 브레이크아웃 등)
    # 여기서는 touches, side, symbol, message, center 가정. 없으면 message에서 center 파싱됨.
    need_cols = ["symbol", "ts", "side", "touches", "message"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = ""
    return df.dropna(subset=["symbol", "ts"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--procs", type=int, default=max(2, os.cpu_count() // 2))
    ap.add_argument("--max-fetch-concurrency", type=int, default=2, help="동시 OHLCV 호출 상한")
    ap.add_argument("--http-retries", type=int, default=3)
    ap.add_argument("--http-backoff", type=float, default=0.8)
    ap.add_argument("--touches", type=int, choices=[2,3], default=2, help="필터 기준(2 or 3 touches)")
    args = ap.parse_args()

    df_sig = load_tv_signals(args.signals)

    # 심볼 목록
    symbols = sorted(df_sig["symbol"].dropna().unique().tolist())
    print(f"[BT] starting with {len(symbols)} symbols using {args.procs} procs...")

    # 프로세스 풀 + 세마포어(동시 fetch 제한)
    mp = get_context("spawn")
    sem = mp.BoundedSemaphore(value=max(1, args.max_fetch_concurrency))
    pool = mp.Pool(
        processes=max(1, args.procs),
        initializer=set_throttle,
        initargs=(sem, args.http_retries, args.http_backoff),
        maxtasksperchild=50,  # 메모리 누수 방지
    )

    tasks = []
    for sym in symbols:
        rows = df_sig[df_sig["symbol"] == sym].copy()
        tasks.append((sym, args.timeframe, rows, args.touches))

    try:
        parts = pool.map(worker, tasks)
    except KeyboardInterrupt:
        print("\n[BT] interrupted by user, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)
    except Exception:
        pool.terminate()
        pool.join()
        raise
    finally:
        pool.close()
        pool.join()

    # 결과 취합
    ts_list = []
    st_list = []
    for tdf, sdf in parts:
        if tdf is not None and not tdf.empty:
            ts_list.append(tdf)
        if sdf is not None and not sdf.empty:
            st_list.append(sdf)

    trades = pd.concat(ts_list, ignore_index=True) if ts_list else pd.DataFrame()
    stats  = pd.concat(st_list, ignore_index=True) if st_list else pd.DataFrame()

    out_trades = "./logs/bt_tv_breakout_trades.csv"
    out_stats  = "./logs/bt_tv_breakout_stats.csv"
    os.makedirs("./logs", exist_ok=True)
    trades.to_csv(out_trades, index=False)
    stats.to_csv(out_stats, index=False)

    if not trades.empty:
        print(f"\n=== TV Breakout (close-based / 24h expiry / fees=0.1% RT) ===")
        print(f"Trades saved: {out_trades} (rows={len(trades)})")
        print(f"Stats  saved: {out_stats} (rows={len(stats)})\n")
        # 간단 집계
        g = stats.agg({"trades":"sum","win_rate":"mean","avg_net":"mean","median_net":"mean","total_net":"sum"})
        print(g.to_frame(name="overall").T.to_string(index=False))
    else:
        print("No trades generated.")

if __name__ == "__main__":
    main()