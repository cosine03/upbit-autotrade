# -*- coding: utf-8 -*-
"""
backtest_tv_events_mp.py
- TV 이벤트(Detected / Price in Box / Box Breakout / Line Breakout) 백테스트 (멀티프로세싱, Windows 호환)
- 로컬 CSV OHLCV만 사용 (외부 의존 X)
- 옵션: --ohlcv-roots / --ohlcv-patterns / --assume-ohlcv-tz
"""

import os, argparse, math, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, freeze_support, cpu_count

# ---------- 공통 유틸 ----------
def to_utc_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
    return pd.to_datetime(x, utc=True, errors="coerce")

def to_ns(s: pd.Series) -> np.ndarray:
    s = pd.to_datetime(s, utc=True, errors="coerce")
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]").astype("int64").to_numpy()

def parse_expiries(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok: continue
        if tok.endswith("h"): out.append(float(tok[:-1]))
        elif tok.endswith("m"): out.append(float(tok[:-1])/60.0)
        else: out.append(float(tok))
    return out

def floor_index(ts64: np.ndarray, key_ns: int) -> int:
    i = int(np.searchsorted(ts64, key_ns, side="right")) - 1
    return i

def load_ohlcv_local(symbol: str, timeframe: str, roots: list, patterns: list, assume_tz: str|None):
    for r in roots:
        for p in patterns:
            rel = p.format(symbol=symbol, timeframe=timeframe)
            path = os.path.join(r, rel) if not (rel.startswith("./") or rel.startswith(".\\")) else rel
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    # 컬럼 표준화
                    cols = {c.lower(): c for c in df.columns}
                    # ts 확보
                    if "ts" in cols:
                        ts = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")
                    elif "timestamp" in cols:
                        ts = pd.to_datetime(df[cols["timestamp"]], utc=True, errors="coerce")
                    else:
                        # 인덱스가 시간일 수도
                        if isinstance(df.index, pd.DatetimeIndex):
                            ts = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
                        else:
                            continue
                    out = pd.DataFrame({
                        "ts": ts,
                        "open": pd.to_numeric(df.get("open") or df.get(cols.get("open","open")), errors="coerce"),
                        "high": pd.to_numeric(df.get("high") or df.get(cols.get("high","high")), errors="coerce"),
                        "low":  pd.to_numeric(df.get("low")  or df.get(cols.get("low","low")), errors="coerce"),
                        "close":pd.to_numeric(df.get("close")or df.get(cols.get("close","close")), errors="coerce"),
                        "volume":pd.to_numeric(df.get("volume")or df.get(cols.get("volume","volume")), errors="coerce"),
                    }).dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)
                    if assume_tz and getattr(out["ts"].dtype, "tz", None) is None:
                        # 입력이 naive면 지정 tz로 가정 후 UTC로 변환
                        out["ts"] = pd.to_datetime(out["ts"]).dt.tz_localize(assume_tz).dt.tz_convert("UTC")
                    return out
                except Exception:
                    continue
    return pd.DataFrame()

# ---------- 시뮬 로직 ----------
def simulate_symbol(symbol: str, ohlcv: pd.DataFrame, sig_rows: pd.DataFrame,
                    timeframe: str, tp: float, sl: float, fee_rt: float, expiry_h: float):
    if ohlcv.empty or sig_rows.empty:
        return pd.DataFrame()
    ts64 = to_ns(ohlcv["ts"])
    out = []
    for _, r in sig_rows.iterrows():
        sig_ts = to_utc_ts(r["ts"])
        i = floor_index(ts64, int(sig_ts.value))
        if i < 0 or i >= len(ohlcv): continue
        entry = float(ohlcv["close"].iloc[i])
        side  = str(r["side"]).lower()
        if side not in ("support","resistance"):  # 메시지에서 추출 실패 시 est_level로 방향 추정
            side = "support" if float(r.get("est_level", np.nan)) <= entry else "resistance"
        # TP/SL 절대값
        tp_abs = entry * (1 + (tp/100.0)) if side=="resistance" else entry * (1 - (tp/100.0))
        sl_abs = entry * (1 - (sl/100.0)) if side=="resistance" else entry * (1 + (sl/100.0))
        # 만기 인덱스
        max_ahead = int(math.ceil((expiry_h*60)/15.0))  # 15m 기준
        hit = None
        for j in range(i+1, min(i+1+max_ahead, len(ohlcv))):
            hi = float(ohlcv["high"].iloc[j])
            lo = float(ohlcv["low"].iloc[j])
            if side=="resistance":
                if hi >= tp_abs: hit = ("tp", j); break
                if lo <= sl_abs: hit = ("sl", j); break
            else:  # support
                if lo <= tp_abs: hit = ("tp", j); break
                if hi >= sl_abs: hit = ("sl", j); break
        if hit:
            kind, j = hit
            gross = (tp/100.0) if kind=="tp" else -(sl/100.0)
        else:
            j = min(i+max_ahead, len(ohlcv)-1)
            exit_px = float(ohlcv["close"].iloc[j])
            gross = (exit_px - entry)/entry if side=="resistance" else (entry - exit_px)/entry
        net = gross - fee_rt
        out.append({
            "symbol": symbol, "ts": sig_ts, "strategy": f"{r['event']}_{expiry_h}h_{tp}/{sl}",
            "entry": entry, "gross": gross, "net": net, "side": side, "expiry_h": expiry_h
        })
    return pd.DataFrame(out)

# ---------- 워커 ----------
def worker(task: dict):
    symbol = task["symbol"]
    ohlcv = load_ohlcv_local(symbol, task["timeframe"], task["roots"], task["patterns"], task["assume_tz"])
    if ohlcv.empty:
        print(f"[{symbol}] get_ohlcv returned empty.")
        return (pd.DataFrame(), pd.DataFrame())
    trades_all = []
    for exp_h in task["expiries"]:
        tr = simulate_symbol(symbol, ohlcv, task["rows"], task["timeframe"],
                             task["tp"], task["sl"], task["fee"], exp_h)
        if not tr.empty:
            tr["group"] = task["group"]
            trades_all.append(tr)
    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    # 통계
    if trades.empty:
        return (pd.DataFrame(), pd.DataFrame())
    def agg_stats(g):
        return pd.Series({
            "trades": len(g),
            "win_rate": (g["net"]>0).mean() if len(g) else 0.0,
            "avg_net": g["net"].mean() if len(g) else 0.0,
            "median_net": g["net"].median() if len(g) else 0.0,
            "total_net": g["net"].sum() if len(g) else 0.0,
        })
    stats = trades.groupby(["group","expiry_h"], as_index=False).apply(agg_stats).reset_index(drop=True)
    return (trades, stats)

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiries", default="4h,8h")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--procs", type=int, default=cpu_count())
    ap.add_argument("--dist-max", type=float, default=None, help="distance_pct 상한(%)")
    ap.add_argument("--ohlcv-roots", default=".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv")
    ap.add_argument("--ohlcv-patterns", default="data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;ohlcv/{symbol}-{timeframe}.csv;ohlcv/{symbol}_{timeframe}.csv;logs/ohlcv/{symbol}-{timeframe}.csv;logs/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv")
    ap.add_argument("--assume-ohlcv-tz", default=None, help="입력 CSV가 naive면 이 tz로 간주 후 UTC로 변환 (예: 'UTC' 또는 'Asia/Seoul')")
    ap.add_argument("--outdir", default="./logs")
    args = ap.parse_args()

    print("[BT] BACKTEST TV EVENTS (LOCAL_SIM v3, no external deps)")
    df = pd.read_csv(args.signals)
    if "ts" not in df.columns: raise ValueError("signals 파일에 'ts' 필요")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "timeframe" not in df.columns: df["timeframe"] = args.timeframe
    if args.dist_max is not None:
        before = len(df)
        df = df[df["distance_pct"].astype(float) <= float(args.dist_max*100 if args.dist_max <= 1 else args.dist_max)]
        print(f"[BT] distance_pct filter {args.dist_max}: {before}->{len(df)} rows")

    groups = {
        "detected": df[df["event"].str.contains("detected", case=False, na=False)],
        "price_in_box": df[df["event"].str.contains("price_in_box", case=False, na=False)],
        "box_breakout": df[df["event"].str.contains("box_breakout", case=False, na=False)],
        "line_breakout": df[df["event"].str.contains("line_breakout", case=False, na=False)],
    }

    roots = [p.strip() for p in args.ohlcv_roots.split(";") if p.strip()]
    patterns = [p.strip() for p in args.ohlcv_patterns.split(";") if p.strip()]
    expiries_h = parse_expiries(args.expiries)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[BT] signals rows={len(df)}, symbols={df['symbol'].nunique()}, timeframe={args.timeframe}")

    all_trades, all_stats = [], []
    for grp_name, gdf in groups.items():
        if gdf.empty:
            print(f"[BT][{grp_name}] no tasks.")
            continue
        sym_list = sorted(gdf["symbol"].unique().tolist())
        tasks = []
        for sym in sym_list:
            rows = gdf[gdf["symbol"]==sym].copy()
            tasks.append({
                "symbol": sym, "rows": rows, "group": grp_name,
                "timeframe": args.timeframe, "tp": args.tp, "sl": args.sl, "fee": args.fee,
                "expiries": expiries_h, "roots": roots, "patterns": patterns, "assume_tz": args.assume_ohlcv_tz
            })
        print(f"[BT][{grp_name}] start: symbols={len(sym_list)} rows={len(gdf)} tasks={len(tasks)} procs={args.procs}")
        with Pool(processes=args.procs) as pool:
            parts = list(pool.imap_unordered(worker, tasks, chunksize=max(1, len(tasks)//(args.procs*2) or 1)))
        trades = pd.concat([p[0] for p in parts if p and not p[0].empty], ignore_index=True) if parts else pd.DataFrame()
        stats  = pd.concat([p[1] for p in parts if p and not p[1].empty], ignore_index=True) if parts else pd.DataFrame()
        if not trades.empty:
            trades.to_csv(os.path.join(args.outdir, f"bt_tv_events_trades_{grp_name}.csv"), index=False)
            stats.to_csv(os.path.join(args.outdir, f"bt_tv_events_stats_{grp_name}.csv"), index=False)
            print(f"[BT][{grp_name}] trades -> {os.path.join(args.outdir, f'bt_tv_events_trades_{grp_name}.csv')} (rows={len(trades)})")
            print(f"[BT][{grp_name}] stats  -> {os.path.join(args.outdir, f'bt_tv_events_stats_{grp_name}.csv')} (rows={len(stats)})")
            all_trades.append(trades); all_stats.append(stats)

    if all_trades:
        trades_all = pd.concat(all_trades, ignore_index=True)
        stats_all  = pd.concat(all_stats, ignore_index=True)
        # 요약
        summary = stats_all.groupby(["group","expiry_h"], as_index=False).agg({
            "trades":"sum","win_rate":"mean","avg_net":"mean","median_net":"mean","total_net":"sum"
        })
        print("\n=== Summary (by event group & expiry) ===")
        print(summary.to_string(index=False))
        trades_all.to_csv(os.path.join(args.outdir, "bt_tv_events_trades.csv"), index=False)
        stats_all.to_csv(os.path.join(args.outdir, "bt_tv_events_stats.csv"), index=False)
        print(f"\n[BT] saved -> {os.path.join(args.outdir,'bt_tv_events_trades.csv')} (rows={len(trades_all)})")
        print(f"[BT] saved -> {os.path.join(args.outdir,'bt_tv_events_stats.csv')} (rows={len(stats_all)})")
    else:
        pd.DataFrame().to_csv(os.path.join(args.outdir, "bt_tv_events_stats.csv"), index=False)
        print(f"[BT] saved -> {os.path.join(args.outdir,'bt_tv_events_stats.csv')} (rows=0)")

if __name__ == "__main__":
    freeze_support()
    main()