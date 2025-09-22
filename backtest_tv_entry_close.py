# -*- coding: utf-8 -*-
"""
backtest_tv_entry_close.py
- TradingView(폴 지표) 알람 기반 백테스트
- 진입: 신호가 나온 '그 봉'의 종가에 Long 진입
- 만기: 4h / 8h (기본), 봉단위로 15m 기준 16/32바
- 방향: 롱온리 (기본은 side=='support'만 사용, --sides any 로 해제 가능)
- 수수료: 왕복 0.1% (각 0.05%)

입력 CSV 예 (signals_tv.csv):
ts,event,side,level,touches,symbol,timeframe,extra,source,host,message
2025-09-20T07:45:06+00:00,price_in_box,resistance,2,2,KRW-LINK,,TV,52.32.178.7,Price in Box ...

사용법:
python backtest_tv_entry_close.py ./logs/signals_tv.csv
python backtest_tv_entry_close.py ./logs/signals_tv.csv --timeframe 15m --expiry 4h,8h --sides support
"""
import os
import argparse
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

# upbit ohlcv 로딩은 sr_engine 경유 (실시간 스크립트와 동일 계열)
from sr_engine.data import get_ohlcv

# ------------------ 유틸 ------------------
def ensure_ts_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
            ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
            df["ts"] = ts
        else:
            for c in ("timestamp", "time", "datetime", "date"):
                if c in df.columns:
                    df["ts"] = pd.to_datetime(df[c], errors="coerce", utc=True)
                    break
        if "ts" not in df.columns:
            # 최후의 보정(분 단위 증가)
            now = pd.Timestamp.utcnow().tz_localize("UTC")
            df["ts"] = pd.date_range(end=now, periods=len(df), freq="T", tz="UTC")
    df = df.sort_values("ts").reset_index(drop=True)
    df = df[~df["ts"].isna()].reset_index(drop=True)
    return df

def tf_minutes(tf: str) -> int:
    s = (tf or "15m").strip().lower()
    if s.endswith("m"): return int(s[:-1])
    if s.endswith("h"): return int(s[:-1]) * 60
    if s.endswith("d"): return int(s[:-1]) * 60 * 24
    return 15

def expiry_bars(expiry: str, timeframe: str) -> int:
    m = tf_minutes(timeframe)
    if expiry.lower().endswith("h"):
        hours = float(expiry[:-1])
        return int(round((hours * 60) / m))
    if expiry.lower().endswith("m"):
        mins = float(expiry[:-1])
        return int(round(mins / m))
    # 기본 4h
    return int(round((4 * 60) / m))

def load_signals_tv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ts 파싱 (mixed/ISO 모두 허용)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    # 심볼 정규화
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    else:
        df["symbol"] = ""
    # 기본 컬럼 채우기
    for col in ["event", "side", "timeframe"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    return df

# CSV 캐시(파켓 미사용) ---------------------------------------------------
CACHE_DIR = "./ohlcv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_ohlcv_cached(symbol: str, timeframe: str = "15m") -> pd.DataFrame:
    fn = f"{symbol.replace('-', '_')}_{timeframe}.csv"
    path = os.path.join(CACHE_DIR, fn)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"])
            return df
        except Exception:
            pass
    df = get_ohlcv(symbol, timeframe)
    df = ensure_ts_column(df)
    # 저장용: 표준 컬럼만
    keep = [c for c in ["ts","open","high","low","close","volume","value"] if c in df.columns]
    df[keep].to_csv(path, index=False)
    return df[keep]

# ------------------ 진입/청산 로직 ------------------
def bar_index_for_signal_close_entry(ohlcv: pd.DataFrame, signal_ts_utc: pd.Timestamp) -> int:
    """
    '신호가 발생한 그 봉'을 찾아 그 봉의 종가에 진입.
    ohlcv['ts']는 봉 시작 시각이라 가정. signal_ts ∈ [ts[i], ts[i+1]) 이면 i번째 종가 진입.
    """
    ts = ohlcv["ts"].to_numpy()
    # numpy searchsorted는 tz-naive를 원하므로, ns값으로 비교
    ts_ns = ts.astype("datetime64[ns]")
    idx = np.searchsorted(ts_ns, np.datetime64(signal_ts_utc.to_datetime64()), side="right") - 1
    return int(max(0, min(idx, len(ohlcv)-1)))

def simulate_long_from_bar(ohlcv: pd.DataFrame, entry_i: int, tp_pct: float, sl_pct: float, expiry_bars_: int, fee_rtrip: float = 0.001) -> Dict:
    """
    - entry_i: 진입 바 인덱스 (그 바의 'close'에 진입)
    - 이후 바들에서 TP/SL 체결 탐지(보수적으로 같은 바에 TP/SL 동시 히트 시 SL 우선)
    - 만기 시 close로 청산
    반환: dict(result='tp/sl/expiry', net_pct, hold_bars, exit_i)
    """
    entry_i = int(entry_i)
    if entry_i >= len(ohlcv):
        return {"result": "invalid", "net_pct": 0.0, "hold_bars": 0, "exit_i": entry_i}

    entry_px = float(ohlcv["close"].iloc[entry_i])
    tp_px = entry_px * (1.0 + tp_pct/100.0)
    sl_px = entry_px * (1.0 - sl_pct/100.0)

    # 다음 바부터 체크
    end_i = min(len(ohlcv)-1, entry_i + expiry_bars_)
    for i in range(entry_i + 1, end_i + 1):
        hi = float(ohlcv["high"].iloc[i])
        lo = float(ohlcv["low"].iloc[i])
        # 보수적: SL 충족 시 우선
        if lo <= sl_px:
            net = -(sl_pct/100.0) - fee_rtrip
            return {"result": "sl", "net_pct": net, "hold_bars": i - entry_i, "exit_i": i}
        if hi >= tp_px:
            net = (tp_pct/100.0) - fee_rtrip
            return {"result": "tp", "net_pct": net, "hold_bars": i - entry_i, "exit_i": i}

    # 만기 청산
    close_px = float(ohlcv["close"].iloc[end_i])
    net = (close_px / entry_px - 1.0) - fee_rtrip
    return {"result": "expiry", "net_pct": net, "hold_bars": end_i - entry_i, "exit_i": end_i}

# ------------------ 메인 루틴 ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("signals_csv", help="TV 알람 CSV (예: ./logs/signals_tv.csv)")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--expiry", default="4h,8h", help="쉼표로 여러개 지정 (예: 4h,8h)")
    parser.add_argument("--sides", default="support", choices=["support","any"], help="롱온리 기준 side 필터 (기본 support, any면 전체)")
    parser.add_argument("--fee_roundtrip", type=float, default=0.001, help="왕복 수수료(기본 0.001=0.1%)")
    # 전략 세트(기본 6개)
    parser.add_argument("--strategies", default="stable_1.5/1.0,aggressive_2.0/1.25,scalp_1.0/0.75,mid_1.25/1.0,mid_1.75/1.25,tight_0.8/0.8",
                        help="이름_tp/sl 퍼센트. 예: name_1.5/1.0 여러개 콤마")
    parser.add_argument("--out_trades", default="./logs/bt_tv_entryclose_trades.csv")
    parser.add_argument("--out_stats", default="./logs/bt_tv_entryclose_stats.csv")
    args = parser.parse_args()

    df = load_signals_tv(args.signals_csv)

    # 사이드 필터(롱온리): 기본 support만
    if args.sides == "support":
        df = df[df["side"].str.lower().eq("support")]

    # 심볼/타임프레임 정리
    df["symbol"] = df["symbol"].str.upper()
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    if not symbols:
        print("[BT] No symbols found in signals_tv.csv")
        return

    # OHLCV 미리 로드/캐시
    ohlcvs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            ohl = get_ohlcv_cached(sym, args.timeframe)
            ohl = ensure_ts_column(ohl)
            ohlcvs[sym] = ohl
        except Exception as ex:
            print(f"[BT] OHLCV load failed: {sym} -> {ex}")

    # 전략 파싱
    strategies: List[Tuple[str, float, float]] = []
    for token in [t.strip() for t in args.strategies.split(",") if t.strip()]:
        # name_tp/sl
        # e.g., stable_1.5/1.0
        try:
            name, pair = token.split("_", 1)
            tp_s, sl_s = pair.split("/", 1)
            strategies.append((name, float(tp_s), float(sl_s)))
        except Exception:
            print(f"[BT] skip invalid strategy format: {token}")

    # 만기 파싱
    expiries = [e.strip() for e in args.expiry.split(",") if e.strip()]
    exp_bars_map = {e: expiry_bars(e, args.timeframe) for e in expiries}

    trades_rows = []
    stats_rows = []

    for strat_name, tp, sl in strategies:
        total_trades = 0
        for sym in symbols:
            if sym not in ohlcvs:
                continue
            ohl = ohlcvs[sym]
            ts_arr = ohl["ts"].to_numpy()
            for _, s in df[df["symbol"] == sym].iterrows():
                sig_ts = pd.Timestamp(s["ts"]).tz_convert("UTC") if pd.Timestamp(s["ts"]).tzinfo else pd.Timestamp(s["ts"]).tz_localize("UTC")
                entry_i = bar_index_for_signal_close_entry(ohl, sig_ts)
                for exp_name, exp_n in exp_bars_map.items():
                    res = simulate_long_from_bar(ohl, entry_i, tp, sl, exp_n, fee_rtrip=args.fee_roundtrip)
                    trades_rows.append({
                        "strategy": f"{strat_name}_{tp}/{sl}_{exp_name}",
                        "symbol": sym,
                        "signal_ts": sig_ts.isoformat(),
                        "entry_i": entry_i,
                        "entry_ts": ohl["ts"].iloc[entry_i].isoformat(),
                        "exit_i": res["exit_i"],
                        "exit_ts": ohl["ts"].iloc[res["exit_i"]].isoformat(),
                        "result": res["result"],
                        "net_pct": res["net_pct"],
                        "hold_bars": res["hold_bars"],
                        "tp": tp,
                        "sl": sl,
                        "expiry": exp_name,
                    })
                    total_trades += 1

        print(f"[BT] {strat_name} ({tp}/{sl}) done. symbols={len(symbols)} trades={total_trades}")

    # 결과 집계
    trades_df = pd.DataFrame(trades_rows)
    if len(trades_df) == 0:
        print("[BT] No trades simulated.")
        return

    # 전략별 통계
    def agg_stats(g):
        wins = (g["result"] == "tp").sum()
        wrate = wins / len(g) if len(g) else 0.0
        return pd.Series({
            "trades": len(g),
            "win_rate": wrate,
            "avg_net": g["net_pct"].mean(),
            "median_net": g["net_pct"].median(),
            "total_net": g["net_pct"].sum(),
        })

    stats_df = trades_df.groupby("strategy", as_index=False).apply(agg_stats)

    os.makedirs(os.path.dirname(args.out_trades) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_stats) or ".", exist_ok=True)
    trades_df.to_csv(args.out_trades, index=False)
    stats_df.to_csv(args.out_stats, index=False)

    print("\n=== TV Backtest (Entry = signal-bar close / Long-only / fees=0.1% roundtrip) ===")
    print(f"Trades saved: {args.out_trades} (rows={len(trades_df)})")
    print(f"Stats  saved: {args.out_stats} (rows={len(stats_df)})\n")
    print(stats_df.sort_values("avg_net", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
