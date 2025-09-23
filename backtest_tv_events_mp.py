# -*- coding: utf-8 -*-
"""
TV 이벤트(backtest) 멀티프로세싱 버전 (안정화)

- 이벤트 그룹별( detected / price_in_box / box_breakout / line_breakout ) 롱 온리
- 엔트리: '신호봉 다음 봉의 종가'가 (저항 기준) 레벨 위에서 마감하면 롱 진입
  * support 쪽 신호는 롱에서 무시(필요시 옵션화 가능)
- TP/SL: 퍼센트 (기본 TP 1.5%, SL 1.25%), 수수료 왕복 fee(기본 0.1%) 적용
- 만기: 다중 만기 (기본 4h,8h)
- signals_tv_enriched.csv에 존재 가능: distance_pct → --dist-max 로 필터
- 빈 OHLCV나 tz 문제, 피클링 에러(워커 반환 시) 방어

예시 실행(Windows PowerShell):
python backtest_tv_events_mp.py .\logs\signals_tv.csv ^
  --timeframe 15m ^
  --expiries 4h,8h ^
  --tp 1.5 ^
  --sl 1.25 ^
  --fee 0.001 ^
  --dist-max 0.25 ^
  --procs 24
"""

import os, sys, csv, math, uuid, tempfile, traceback
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from multiprocessing import Pool, get_start_method, set_start_method

# ---- 프로젝트 함수 로드 ----
try:
    from sr_engine.data import get_ohlcv
except Exception:
    get_ohlcv = None

# ---------- 유틸 ----------
def parse_expiries(s: str) -> List[float]:
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip().lower()
        if not tok: 
            continue
        if tok.endswith("h"):
            out.append(float(tok[:-1]))
        elif tok.endswith("m"):
            out.append(float(tok[:-1]) / 60.0)
        else:
            out.append(float(tok))
    return out or [4.0, 8.0]

def tf_minutes(tf: str) -> int:
    s = tf.strip().lower()
    if s.endswith("m"): return int(s[:-1])
    if s.endswith("h"): return int(s[:-1]) * 60
    if s.endswith("d"): return int(s[:-1]) * 60 * 24
    return 15

def ensure_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV에 ts 컬럼 보장 + UTC tz-aware → np.datetime64 라인에서 비교 안전"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    # ts 확보
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        else:
            # 흔한 후보
            for c in ("time","timestamp","datetime","date"):
                if c in df.columns:
                    ts = pd.to_datetime(df[c], errors="coerce", utc=True)
                    break
            else:
                # 마지막 수단: 인덱스를 시간으로 간주
                ts = pd.to_datetime(df.index, errors="coerce", utc=True)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        df["ts"] = ts
    else:
        ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        df["ts"] = ts
    # 정렬/정리
    keep_cols = [c for c in ["ts","open","high","low","close","volume"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["ts","open","high","low","close"]).sort_values("ts")
    df = df.reset_index(drop=True)
    return df

def ts_series_to_ns64(s: pd.Series) -> np.ndarray:
    """UTC-aware 시리즈를 np.datetime64[ns] 배열로."""
    # pandas 2.x: tz-aware → .dt.tz_convert('UTC').view('int64') 권장X
    # 안전 경로: astype('datetime64[ns, UTC]') → tz-localize None → ns64
    if hasattr(s, "dt"):
        s2 = pd.to_datetime(s.dt.tz_convert("UTC"), utc=True, errors="coerce")
    else:
        s2 = pd.to_datetime(s, utc=True, errors="coerce")
    s2 = s2.dt.tz_convert("UTC").dt.tz_localize(None)
    return s2.astype("datetime64[ns]").to_numpy()

def to_npdt64(t: pd.Timestamp) -> np.datetime64:
    # tz-aware → UTC → naive → to numpy datetime64
    if not isinstance(t, pd.Timestamp):
        t = pd.Timestamp(t, utc=True)
    else:
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
    t = t.tz_localize(None)
    return t.to_datetime64()

def fee_roundtrip(pct: float) -> float:
    # 왕복 수수료 pct → 단순 차감용
    return max(0.0, float(pct))

# ---------- OHLCV 로드 ----------
def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    try:
        if get_ohlcv is None:
            raise RuntimeError("sr_engine.data.get_ohlcv not available")
        raw = get_ohlcv(symbol, timeframe)
        if raw is None or len(raw) == 0:
            print(f"[{symbol}] get_ohlcv returned empty.")
            return pd.DataFrame()
        df = raw.reset_index(drop=False) if isinstance(raw.index, pd.DatetimeIndex) and "ts" not in raw.columns else raw.copy()
        df = ensure_ts_col(df)
        return df
    except Exception as e:
        print(f"[{symbol}] OHLCV load error: {repr(e)}")
        return pd.DataFrame()

# ---------- 엔트리 판단 (롱/저항만) ----------
def breakout_close_entry(ohlcv: pd.DataFrame, sig_ts: pd.Timestamp, level_price: Optional[float]) -> Optional[Tuple[int, float]]:
    """
    신호봉 다음 봉의 '종가'가 level_price 위에서 마감하면 엔트리.
    level_price가 None이면 신호 시점 종가를 레벨 대용으로 사용(근사).
    반환: (엔트리 인덱스, 엔트리 가격) or None
    """
    if ohlcv.empty: 
        return None
    ts64 = ts_series_to_ns64(ohlcv["ts"])
    # 신호봉 인덱스(소속 봉)
    key64 = to_npdt64(sig_ts)
    i_sig = int(np.searchsorted(ts64, key64, side="right")) - 1
    if i_sig < 0 or i_sig+1 >= len(ohlcv):
        return None
    # 다음 봉 종가가 레벨 위?
    close_next = float(ohlcv["close"].iloc[i_sig+1])
    ref_level = float(level_price) if level_price is not None and np.isfinite(level_price) else float(ohlcv["close"].iloc[i_sig])
    if close_next > ref_level:
        return (i_sig+1, close_next)
    return None

# ---------- 포지션 시뮬 ----------
def simulate_rows_for_symbol(symbol: str,
                             df_sig: pd.DataFrame,
                             ohlcv: pd.DataFrame,
                             tp_pct: float,
                             sl_pct: float,
                             fee_rt: float,
                             expiries_h: List[float]) -> List[Dict]:
    """
    df_sig: 해당 심볼+이벤트 그룹에 속하는 row들 (ts, event, side, level, touches, message, level_estimate?)
    롱온리: side=='resistance' 만 사용.
    TP/SL는 엔트리 기준 퍼센트.
    만기 리스트 각각 평가.
    """
    out: List[Dict] = []
    if ohlcv.empty or df_sig.empty:
        return out

    # 오프셋 캐시
    ts_arr = ohlcv["ts"].to_numpy()
    ts64 = ts_series_to_ns64(ohlcv["ts"])
    opens = ohlcv["open"].to_numpy()
    highs = ohlcv["high"].to_numpy()
    lows  = ohlcv["low"].to_numpy()
    closes= ohlcv["close"].to_numpy()

    bar_minutes = tf_minutes("15m")  # 실행 CLI에서 timeframe=15m만 쓰는 중이라면 고정도 OK. 원하면 파라미터로 교체.
    # 정확히 하려면 timeframe 인수로 받아도 됨.

    for s in df_sig.itertuples():
        # side 필터(롱=저항만)
        if str(s.side).lower() != "resistance":
            continue
        # 신호 시각 UTC
        sig_ts = pd.to_datetime(getattr(s, "ts"), utc=True, errors="coerce")
        if sig_ts is pd.NaT:
            continue

        # level_estimate(or None)
        level_est = getattr(s, "level_estimate", None)
        try:
            level_est = float(level_est) if level_est is not None and str(level_est) not in ("", "nan") else None
        except:
            level_est = None

        entry = breakout_close_entry(ohlcv, sig_ts, level_est)
        if entry is None:
            continue
        i_ent, px_ent = entry

        # TP/SL 절대가
        tp_price = px_ent * (1.0 + tp_pct/100.0)
        sl_price = px_ent * (1.0 - sl_pct/100.0)

        # 각 만기별 평가
        for eh in expiries_h:
            end_ts = sig_ts + pd.Timedelta(hours=float(eh))
            end64  = to_npdt64(end_ts)
            # 엔트리 이후 ~ 만기 전 범위
            i_end = int(np.searchsorted(ts64, end64, side="left"))
            if i_end <= i_ent:
                continue
            # 고저 터치로 TP/SL 순서 판단
            # 같은 봉에서 동시 체결 모호성 → SL 우선(보수적) 또는 high/low 순서로 로직
            hit = None
            for i in range(i_ent+1, i_end+1):
                h = float(highs[i]); l = float(lows[i])
                if l <= sl_price:  # SL first
                    hit = ("SL", i)
                    break
                if h >= tp_price:
                    hit = ("TP", i)
                    break
            if hit is None:
                # 만기 종가 청산
                px_exit = float(closes[i_end-1])
                label   = "EXP"
            else:
                kind, i_hit = hit
                px_exit = float(opens[i_hit])  # 체결 보수적으로 다음 캔들 시가/혹은 해당 봉 종가 등 정책 가능
                label = kind

            gross = (px_exit - px_ent) / px_ent
            net   = gross - fee_roundtrip(fee_rt)
            out.append({
                "symbol": symbol,
                "event": getattr(s, "event", ""),
                "side": getattr(s, "side", ""),
                "level": int(getattr(s, "level", 0) or 0),
                "touches": int(getattr(s, "touches", 0) or 0),
                "ts": str(sig_ts.isoformat()),
                "entry_idx": int(i_ent),
                "entry_px": float(px_ent),
                "exit_label": label,
                "exit_idx": int(i_end if hit is None else i_hit),
                "exit_px": float(px_exit),
                "tp_pct": float(tp_pct),
                "sl_pct": float(sl_pct),
                "expiry_h": float(eh),
                "fee_rt": float(fee_rt),
                "net": float(net),
            })
    return out

# ---------- 워커(파일로 반환) ----------
def worker_simulate_to_csv(tmp_dir: str,
                           symbol: str,
                           timeframe: str,
                           rows_for_symbol: pd.DataFrame,
                           tp_pct: float,
                           sl_pct: float,
                           fee_rt: float,
                           expiries_h: List[float]) -> Optional[str]:
    try:
        ohlcv = load_ohlcv(symbol, timeframe)
        trades = simulate_rows_for_symbol(symbol, rows_for_symbol, ohlcv, tp_pct, sl_pct, fee_rt, expiries_h)
        if not trades:
            return None
        df = pd.DataFrame(trades)
        # numpy dtype → 표준화
        for c in df.columns:
            if df[c].dtype == "object":
                continue
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype(float) if pd.api.types.is_float_dtype(df[c]) else df[c].astype(int)
        path = os.path.join(tmp_dir, f"{symbol}_{uuid.uuid4().hex}.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except Exception as e:
        # 워커 예외도 파일로 남김
        err_path = os.path.join(tmp_dir, f"ERR_{symbol}_{uuid.uuid4().hex}.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(repr(e) + "\n" + traceback.format_exc())
        return None

# ---------- 그룹 실행 ----------
def run_group(dfg: pd.DataFrame, group_name: str, timeframe: str,
              tp_pct: float, sl_pct: float, fee_rt: float, expiries_h: List[float],
              procs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    syms = sorted(dfg["symbol"].unique().tolist())
    print(f"[BT][{group_name}] start: symbols={len(syms)} rows={len(dfg)} tasks={len(syms)} procs={procs}")

    tmp_dir = tempfile.mkdtemp(prefix=f"bt_{group_name}_")
    tasks = []
    for sym in syms:
        rows = dfg[dfg["symbol"] == sym].copy()
        tasks.append((tmp_dir, sym, timeframe, rows, tp_pct, sl_pct, fee_rt, expiries_h))

    paths = []
    if procs and procs > 1:
        # Windows: spawn
        if get_start_method(allow_none=True) != "spawn":
            try:
                set_start_method("spawn", force=True)
            except RuntimeError:
                pass
        with Pool(processes=procs) as pool:
            paths = pool.starmap(worker_simulate_to_csv, tasks)
    else:
        for t in tasks:
            paths.append(worker_simulate_to_csv(*t))

    # 결과 모으기
    csv_paths = [p for p in paths if p and os.path.isfile(p) and not os.path.basename(p).startswith("ERR_")]
    if not csv_paths:
        print(f"[BT][{group_name}] no trades.")
        return pd.DataFrame(), pd.DataFrame()

    parts = [pd.read_csv(p) for p in csv_paths]
    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # 집계
    def agg(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series({"trades":0, "win_rate":np.nan, "avg_net":np.nan, "median_net":np.nan, "total_net":np.nan})
        wins = (df["net"] > 0).sum()
        return pd.Series({
            "trades": float(len(df)),
            "win_rate": float(wins) / float(len(df)),
            "avg_net": float(df["net"].mean()),
            "median_net": float(df["net"].median()),
            "total_net": float(df["net"].sum()),
        })

    stats = trades.groupby("expiry_h", as_index=False, dropna=False).apply(agg)
    print(f"[BT][{group_name}] trades -> {len(trades)} rows; stats rows={len(stats)}")
    return trades, stats

# ---------- 메인 ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", help="signals_tv.csv (또는 *_enriched.csv)")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.25)   # 요청: SL 1.25로 상향
    ap.add_argument("--fee", type=float, default=0.001) # 왕복 0.1%
    ap.add_argument("--dist-max", type=float, default=None, help="distance_pct <= dist_max 필터(예: 0.25)")
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()

    df = pd.read_csv(args.signals)
    # UTC 파싱
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","event","symbol","side"])
    df["event"] = df["event"].str.strip().str.lower()
    df["side"]  = df["side"].str.strip().str.lower()
    if "touches" in df.columns:
        df["touches"] = pd.to_numeric(df["touches"], errors="coerce").fillna(0).astype(int)
    else:
        df["touches"] = 0

    # 거리 필터 (있을 때만)
    if args.dist_max is not None and "distance_pct" in df.columns:
        df = df[(pd.to_numeric(df["distance_pct"], errors="coerce") <= float(args.dist_max))]
        df = df.reset_index(drop=True)

    # (선택) level_estimate 컬럼 없으면 None으로 추가
    if "level_estimate" not in df.columns:
        df["level_estimate"] = np.nan

    # 심볼/이벤트 분포
    syms = sorted(df["symbol"].unique().tolist())
    print(f"[BT] signals rows={len(df)}, symbols={len(syms)}, timeframe={args.timeframe}")

    # 이벤트 그룹들
    groups = [
        ("detected",     df[df["event"].str.contains("level")]),    # level2_detected, level3_detected 등
        ("price_in_box", df[df["event"]=="price_in_box"]),
        ("box_breakout", df[df["event"]=="box_breakout"]),
        ("line_breakout",df[df["event"]=="line_breakout"]),
    ]

    all_trades = []
    all_stats  = []
    expiries_h = parse_expiries(args.expiries)
    for name, dfg in groups:
        if dfg.empty:
            continue
        tr, st = run_group(dfg, name, args.timeframe, args.tp, args.sl, args.fee, expiries_h, args.procs)
        if not tr.empty:
            tr = tr.copy()
            tr["group"] = name
            all_trades.append(tr)
        if not st.empty:
            st = st.copy()
            st["group"] = name
            all_stats.append(st)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats  = pd.concat(all_stats,  ignore_index=True) if all_stats  else pd.DataFrame()

    # 요약 출력
    if not stats.empty:
        # 그룹+만기별 요약
        cols = ["group","expiry_h","trades","win_rate","avg_net","median_net","total_net"]
        stats = stats[["group","expiry_h","trades","win_rate","avg_net","median_net","total_net"]]
        print("\n=== Summary (by event group & expiry) ===")
        print(stats.sort_values(["group","expiry_h"])[cols].to_string(index=False))

    # 저장
    os.makedirs("./logs", exist_ok=True)
    path_tr = "./logs/bt_tv_events_trades.csv"
    path_st = "./logs/bt_tv_events_stats.csv"
    trades.to_csv(path_tr, index=False, encoding="utf-8")
    stats.to_csv(path_st, index=False, encoding="utf-8")
    print(f"\n[BT] saved -> {path_tr} (rows={len(trades)})")
    print(f"[BT] saved -> {path_st} (rows={len(stats)})")

if __name__ == "__main__":
    main()