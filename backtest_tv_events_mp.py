# -*- coding: utf-8 -*-
"""
TV 이벤트 백테스트 (멀티프로세싱, 롱 온리, close-entry, 4h/8h 등 다중 만기)
- distance_pct 필터 지원(선택)
- tz-aware/naive 혼용 에러 방지 (전 구간 UTC-naive int64 ns 축으로 통일)
- get_ohlcv 이용(가능 시), 실패/빈 DF 시 심볼 건너뜀
- line/box 중복: 같은 ts에선 line_breakout 우선(명시)
- TP/SL: next-bar open 진입, 만기 내 첫 터치 우선, 미충족 시 만기 종가 정산
"""

import argparse, os, time, warnings
import numpy as np
import pandas as pd
import multiprocessing as mp

# ====== 옵션 ======
EVENT_MAP = {
    "level2_detected": "detected",
    "level3_detected": "detected",
    "price_in_box":    "price_in_box",
    "box_breakout":    "box_breakout",
    "line_breakout":   "line_breakout",
}
EVENT_ORDER = ["detected", "price_in_box", "box_breakout", "line_breakout"]  # 요약용 출력 순서

# ====== 안전한 ts 변환 ======
def ts_to_ns(values):
    """
    values(Series/Index/list/ndarray/Scalar) -> int64 ns (UTC 기준, tz-naive)
    """
    s = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(s, pd.Series):
        arr = s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    elif isinstance(s, pd.DatetimeIndex):
        arr = s.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    else:
        s = pd.Series(s)
        arr = s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    return arr.astype("int64")

def minutes_from_tf(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 15

# ====== OHLCV 로더 ======
def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """
    sr_engine.data.get_ohlcv 사용 가능 시 호출.
    결과는 columns= ['ts','open','high','low','close'] 로 정규화.
    실패/빈 데이터면 None 반환.
    """
    try:
        from sr_engine.data import get_ohlcv
    except Exception:
        return None

    try:
        df = get_ohlcv(symbol, timeframe)
        if df is None or len(df) == 0:
            return None

        # ts 컬럼 만들기
        if "ts" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                ts = df.index
            else:
                # 흔치 않지만 timestamp/ time / datetime 후보
                for c in ("timestamp","time","datetime","date"):
                    if c in df.columns:
                        ts = pd.to_datetime(df[c], utc=True, errors="coerce")
                        break
                else:
                    return None
            ts = pd.to_datetime(ts, utc=True, errors="coerce")
            df = df.copy()
            df["ts"] = ts

        keep = ["ts","open","high","low","close"]
        for c in keep:
            if c not in df.columns:
                return None

        # 정렬/정규화
        out = (
            df[keep]
            .dropna(subset=["ts","open","high","low","close"])
            .sort_values("ts")
            .reset_index(drop=True)
        )

        # ts를 tz-aware -> tz-naive로
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)

        return out
    except Exception:
        return None

# ====== 시뮬레이션(심볼 단위) ======
def simulate_symbol(symbol: str,
                    df_sig: pd.DataFrame,
                    timeframe: str,
                    tp: float, sl: float,
                    fee_rt: float,
                    expiries_h: list[float]) -> pd.DataFrame:
    """
    - next-bar open 진입
    - TP/SL 퍼센트는 % 단위(예: 1.5 -> 1.5%)
    - 만기 내 첫 터치 우선(TP가 먼저면 승, SL이 먼저면 패)
    - 미충족 시 만기 종가 정산
    """
    ohlcv = load_ohlcv(symbol, timeframe)
    if ohlcv is None or ohlcv.empty:
        # 심볼 스킵
        return pd.DataFrame(columns=[
            "symbol","ts","event","group","expiry_h","tp","sl","net","win","entry_ts","entry","exit_ts","exit","reason"
        ])

    # ns 축 준비
    ts_ns = ts_to_ns(ohlcv["ts"])
    M = minutes_from_tf(timeframe)

    trades = []
    # 이벤트 시그널 루프 (ts 오름차순 가정)
    df_sig = df_sig.sort_values("ts")
    sig_ns_all = ts_to_ns(df_sig["ts"])

    for (idx, row), sig_ns in zip(df_sig.iterrows(), sig_ns_all):
        # signal bar index (signal 시각 포함하는 봉) -> next-bar open 진입
        i_bar = int(np.searchsorted(ts_ns, sig_ns, side="right")) - 1
        if i_bar < 0 or i_bar+1 >= len(ohlcv):
            continue

        entry_i = i_bar + 1
        entry_ts = ohlcv["ts"].iloc[entry_i]
        entry    = float(ohlcv["open"].iloc[entry_i])

        for eh in expiries_h:
            bars = max(1, int(round(eh * 60 / M)))
            end_i = min(len(ohlcv)-1, entry_i + bars)
            if end_i <= entry_i:
                continue

            tp_px = entry * (1.0 + tp/100.0)
            sl_px = entry * (1.0 - sl/100.0)

            hit = None
            exit_px = float(ohlcv["close"].iloc[end_i])
            exit_ts = ohlcv["ts"].iloc[end_i]
            # 엔트리 다음 봉부터 만기까지 순회
            for j in range(entry_i, end_i+1):
                hi = float(ohlcv["high"].iloc[j])
                lo = float(ohlcv["low"].iloc[j])
                if lo <= sl_px and hi >= tp_px:
                    # 동시터치: 보수적으로 SL 우선(또는 tp 우선으로 바꾸어도 됨)
                    hit = ("SL", sl_px, ohlcv["ts"].iloc[j]); break
                if hi >= tp_px:
                    hit = ("TP", tp_px, ohlcv["ts"].iloc[j]); break
                if lo <= sl_px:
                    hit = ("SL", sl_px, ohlcv["ts"].iloc[j]); break

            if hit is None:
                # 만기 종가 정산
                gross = (exit_px / entry) - 1.0
                net   = gross - fee_rt
                reason = "expiry"
                out_px = exit_px
                out_ts = exit_ts
            else:
                if hit[0] == "TP":
                    gross = (hit[1] / entry) - 1.0
                    net   = gross - fee_rt
                else:
                    gross = (hit[1] / entry) - 1.0
                    net   = gross - fee_rt
                reason = hit[0]
                out_px = hit[1]
                out_ts = hit[2]

            trades.append({
                "symbol": symbol,
                "ts": row["ts"],
                "event": row["event"],
                "group": row["event_group"],
                "expiry_h": eh,
                "tp": tp, "sl": sl,
                "entry_ts": entry_ts,
                "entry": entry,
                "exit_ts": out_ts,
                "exit": out_px,
                "net": net,
                "win": 1 if net > 0 else 0,
                "reason": reason,
            })

    return pd.DataFrame(trades)

# ====== 그룹 실행 ======
def run_group(df_all: pd.DataFrame,
              group: str,
              timeframe: str,
              tp: float, sl: float, fee_rt: float,
              expiries_h: list[float],
              procs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_g = df_all[df_all["event_group"] == group]
    if df_g.empty:
        print(f"[BT][{group}] no tasks.")
        return pd.DataFrame(), pd.DataFrame()

    symbols = df_g["symbol"].unique().tolist()
    tasks = [
        (sym, df_g[df_g["symbol"] == sym], timeframe, tp, sl, fee_rt, expiries_h)
        for sym in symbols
    ]
    print(f"[BT][{group}] start: symbols={len(symbols)} rows={len(df_g)} tasks={len(tasks)} procs={procs}")

    # mp 실행
    with mp.get_context("spawn").Pool(processes=procs) as pool:
        parts = pool.starmap(simulate_symbol, tasks)

    trades = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()

    def agg(gr):
        return pd.Series({
            "trades": len(gr),
            "win_rate": gr["win"].mean() if len(gr) else np.nan,
            "avg_net": gr["net"].mean() if len(gr) else np.nan,
            "median_net": gr["net"].median() if len(gr) else np.nan,
            "total_net": gr["net"].sum() if len(gr) else np.nan,
        })

    stats = trades.groupby(["group","expiry_h"], as_index=False, dropna=False).apply(agg)

    return trades, stats

# ====== 인자/메인 ======
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("signals", help="signals_tv.csv 또는 enriched csv")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--expiries", default="4h,8h", help="예: 4h,8h")
    p.add_argument("--tp", type=float, default=1.5)
    p.add_argument("--sl", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--dist-max", type=float, default=None, help="abs(distance_pct) <= dist_max 로 필터")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--outdir", default="./logs")
    return p.parse_args()

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.signals)
    if "ts" not in df.columns or "event" not in df.columns or "symbol" not in df.columns:
        raise SystemExit("signals csv에 ts/event/symbol 컬럼이 필요합니다.")

    print(f"[BT] signals rows={len(df)}, symbols={df['symbol'].nunique()}, timeframe={args.timeframe}")

    # 1) distance_pct 필터(있을 때만)
    if args.dist_max is not None and "distance_pct" in df.columns:
        before = len(df)
        df = df[df["distance_pct"].abs() <= float(args.dist_max)].copy()
        print(f"[BT] distance_pct filter {args.dist_max}: {before}->{len(df)} rows")

    # 2) 이벤트 그룹 라벨링
    df["event_group"] = df["event"].map(EVENT_MAP).fillna(df["event"])

    # 3) line/box 동시 발생 중복 제거(같은 ts/symbol인 경우 line_breakout 우선)
    df = df.sort_values(["symbol","ts","event_group"])
    mask_both = df["event_group"].isin(["box_breakout","line_breakout"])
    if mask_both.any():
        df["prio"] = np.where(df["event_group"]=="line_breakout", 2, 1)  # line > box
        df = (
            df.sort_values(["symbol","ts","prio"])
              .drop_duplicates(subset=["symbol","ts"], keep="last")
              .drop(columns=["prio"])
        )

    expiries_h = [float(x.strip().lower().replace("h","")) for x in args.expiries.split(",") if x.strip()]

    all_trades, all_stats = [], []
    for grp in EVENT_ORDER:
        tr, st = run_group(df, grp, args.timeframe, args.tp, args.sl, args.fee, expiries_h, args.procs)
        if not tr.empty:
            tr_path = os.path.join(args.outdir, f"bt_tv_events_trades_{grp}.csv")
            st_path = os.path.join(args.outdir, f"bt_tv_events_stats_{grp}.csv")
            tr.to_csv(tr_path, index=False)
            st.to_csv(st_path, index=False)
            print(f"[BT][{grp}] trades -> {tr_path} (rows={len(tr)})")
            print(f"[BT][{grp}] stats  -> {st_path} (rows={len(st)})")
            all_trades.append(tr); all_stats.append(st)
        else:
            print(f"[BT][{grp}] no trades.")

    if all_trades:
        trades = pd.concat(all_trades, ignore_index=True)
        stats  = pd.concat(all_stats,  ignore_index=True)
        trades.to_csv(os.path.join(args.outdir, "bt_tv_events_trades.csv"), index=False)
        stats.to_csv(os.path.join(args.outdir, "bt_tv_events_stats.csv"), index=False)
        print("\n=== Summary (by event group & expiry) ===")
        print(stats)

if __name__ == "__main__":
    main()