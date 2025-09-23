# -*- coding: utf-8 -*-
"""
TV signals 백테스트 (MP, 선택형 진입로직)
- Entry mode:
  * close : 신호봉 종가 즉시 진입
  * dip   : 신호 발생 시점 장중가격 이하로 재진입시 진입
- Long only, 만기 N시간 종가 청산, 왕복 수수료 0.1%(0.05%/0.05%)
- tz/np.datetime64 비교 안전화, OHLCV None 안전화, 멀티프로세싱 안전화

Usage 예:
  python backtest_tv_entry_close_mp.py .\logs\signals_tv.csv --group alt --expiry 4h,8h --procs 8 --entry-mode close
  python backtest_tv_entry_close_mp.py .\logs\signals_tv.csv --group alt --expiry 4h,8h --procs 8 --entry-mode dip

출력:
  ./logs/bt_tv_mp_trades.csv
  ./logs/bt_tv_mp_stats.csv
"""

import os, sys, math, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

# ---- 외부 의존: sr_engine.data.get_ohlcv ----
try:
    from sr_engine.data import get_ohlcv
except Exception:
    print("[WARN] sr_engine.data.get_ohlcv import 실패. 스텁을 사용하려면 코드 수정 필요.", file=sys.stderr)
    raise

LOGS_DIR = "./logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ===== 공통 상수 =====
FEE_RT = 0.001  # round-trip 0.1% = 0.001
ENTRY_SLIPPAGE = 0.0  # 체결가 슬리피지(옵션)

MAJOR = {"KRW-BTC", "KRW-ETH"}

# ===== 유틸 =====
def to_utc_series(s: pd.Series) -> pd.Series:
    """Series(datetime-like) -> tz-aware UTC pandas.Timestamp"""
    if pd.api.types.is_datetime64_any_dtype(s):
        ts = pd.to_datetime(s, utc=True)
    else:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts

def ensure_ohlcv_ts(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV에 ts 컬럼 보장 & UTC 정렬"""
    if df is None or len(df) == 0:
        return None
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            ts = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
            out["ts"] = ts
        else:
            for c in ("timestamp","time","datetime","date"):
                if c in out.columns:
                    out["ts"] = to_utc_series(out[c])
                    break
    if "ts" not in out.columns:
        return None
    out = out.sort_values("ts").reset_index(drop=True)
    out = out[~out["ts"].isna()].reset_index(drop=True)
    # numpy 비교용 캐시
    out["_ts64"] = out["ts"].values.astype("datetime64[ns]")
    return out

def parse_expiries(s: str) -> List[int]:
    """'4h,8h' -> [4,8]"""
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok.endswith("h"):
            out.append(int(tok[:-1]))
        elif tok.isdigit():
            out.append(int(tok))
    return [x for x in out if x > 0]

def price_reached(tp: float, sl: float, entry: float, h: float, l: float) -> Tuple[bool, bool]:
    """해당 봉 고가/저가가 TP/SL에 닿았는지 (롱 기준)"""
    tp_px = entry * (1 + tp/100.0)
    sl_px = entry * (1 - sl/100.0)
    hit_tp = h >= tp_px
    hit_sl = l <= sl_px
    return hit_tp, hit_sl

def pick_first_hit(entry: float, tp: float, sl: float, row: pd.Series) -> Tuple[str, float]:
    """같은 봉에서 TP/SL 동시충족시 보수적으로 SL 우선(또는 규칙 선택 가능)"""
    hit_tp, hit_sl = price_reached(tp, sl, entry, float(row["high"]), float(row["low"]))
    if hit_sl and hit_tp:
        # 동시 터치: 더 보수적인 SL 우선
        exit_px = entry * (1 - sl/100.0)
        return "SL", exit_px
    if hit_tp:
        return "TP", entry * (1 + tp/100.0)
    if hit_sl:
        return "SL", entry * (1 - sl/100.0)
    return "", 0.0

# ===== 엔트리 모드 =====
@dataclass
class Strategy:
    name: str
    tp: float
    sl: float
    expiry_h: int
    entry_mode: str  # "close" or "dip"

def next_bar_open_idx(ohlcv: pd.DataFrame, sig_ts_utc: pd.Timestamp) -> int:
    """
    시그널 시각 바로 다음 봉 오픈 인덱스 (15m 데이터라고 가정)
    내부 비교는 모두 tz-naive numpy datetime64[ns] 로 통일.
    """
    sig64 = np.datetime64(sig_ts_utc.tz_convert("UTC").to_pydatetime())
    ts64 = ohlcv["_ts64"].to_numpy()  # datetime64[ns]
    return int(np.searchsorted(ts64, sig64, side="right"))

def find_intrabar_price_at_signal(ohlcv_1m: pd.DataFrame, sig_ts_utc: pd.Timestamp) -> float:
    """
    알람 '발생 시점'의 장중가격(1분봉) — 신호타임스탬프 분의 종가로 근사.
    """
    if ohlcv_1m is None or len(ohlcv_1m) == 0:
        return np.nan
    # 해당 분(UTC) 매칭
    ts_min = sig_ts_utc.floor("min")
    row = ohlcv_1m[ohlcv_1m["ts"] == ts_min]
    if len(row) == 0:
        # 없으면 바로 이전 1분 사용
        row = ohlcv_1m[ohlcv_1m["ts"] <= ts_min].tail(1)
    if len(row) == 0:
        return np.nan
    return float(row.iloc[-1]["close"])

def simulate_symbol(task: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    각 심볼 단위 작업 (멀티프로세스 맵 대상)
    task: {
       'symbol': str,
       'signals': pd.DataFrame(해당 심볼 행),
       'timeframe': '15m',
       'strategies': List[Strategy]
    }
    """
    sym = task["symbol"]
    sigs = task["signals"].copy()
    timeframe = task["timeframe"]
    strategies: List[Strategy] = task["strategies"]

    # OHLCV 로드
    df = get_ohlcv(sym, timeframe)
    if df is None:
        print(f"[{sym}] OHLCV load error: None")
        return pd.DataFrame(), {"symbol": sym, "trades": 0}
    ohlcv = ensure_ohlcv_ts(df)
    if ohlcv is None:
        print(f"[{sym}] OHLCV load error: ts missing")
        return pd.DataFrame(), {"symbol": sym, "trades": 0}

    # dip 모드 대비: 1분봉 데이터
    need_1m = any(st.entry_mode == "dip" for st in strategies)
    ohlcv_1m = None
    if need_1m:
        try:
            df1 = get_ohlcv(sym, "1m")
            ohlcv_1m = ensure_ohlcv_ts(df1)
        except Exception as ex:
            print(f"[{sym}] 1m load fail: {ex}")

    trades = []

    # 시그널 순회
    for _, s in sigs.iterrows():
        # 시그널 시각 UTC
        try:
            sig_ts = pd.Timestamp(s["ts"], tz="UTC") if pd.Timestamp(s["ts"]).tzinfo is None else pd.Timestamp(s["ts"]).tz_convert("UTC")
        except Exception:
            sig_ts = pd.to_datetime(s["ts"], utc=True)

        # 다음봉 오픈 인덱스(엔트리 판단 시작점: i0)
        i0 = next_bar_open_idx(ohlcv, sig_ts)  # 다음 봉 open
        if i0 >= len(ohlcv):
            continue

        # dip 모드용: ‘알람 발생 시점 장중가격’
        signal_price = np.nan
        if need_1m:
            signal_price = find_intrabar_price_at_signal(ohlcv_1m, sig_ts)

        for st in strategies:
            tp, sl = st.tp, st.sl
            expiry_h = st.expiry_h

            # 엔트리 찾기
            entry_idx = None
            entry_px = None

            if st.entry_mode == "close":
                # 신호봉 '다음 봉’ 시가(or 종가)로 바로 진입 — 여기선 다음 봉 '시가'를 사용
                entry_idx = i0
                entry_px = float(ohlcv.iloc[entry_idx]["open"]) * (1 + ENTRY_SLIPPAGE)
            else:
                # dip: 이후 가격이 signal_price 이하로 내려오는 첫 순간의 봉 '종가'로 진입
                if not np.isfinite(signal_price):
                    continue
                # i0부터 순회하며 해당 봉의 low<=signal_price 이면 dip 충족으로 간주
                for j in range(i0, len(ohlcv)):
                    row = ohlcv.iloc[j]
                    if float(row["low"]) <= signal_price:
                        entry_idx = j
                        # 체결은 이 봉 종가로 간주(보수적)
                        entry_px = float(row["close"]) * (1 + ENTRY_SLIPPAGE)
                        break
                if entry_idx is None:
                    continue  # 진입 못함

            # 만기 인덱스
            dt_entry = ohlcv.iloc[entry_idx]["ts"]
            dt_expiry = dt_entry + pd.Timedelta(hours=expiry_h)
            ts64 = ohlcv["_ts64"].to_numpy()
            exp64 = np.datetime64(dt_expiry.tz_convert("UTC").to_pydatetime())
            idx_exp = int(np.searchsorted(ts64, exp64, side="left"))
            if idx_exp <= entry_idx:
                continue
            idx_exp = min(idx_exp, len(ohlcv)-1)

            # TP/SL/만기 탐색
            exit_reason = "EXP"
            exit_idx = idx_exp
            exit_px = float(ohlcv.iloc[idx_exp]["close"])

            for k in range(entry_idx, idx_exp+1):
                row = ohlcv.iloc[k]
                hit, px = pick_first_hit(entry_px, tp, sl, row)
                if hit:
                    exit_reason = hit
                    exit_idx = k
                    exit_px = px
                    break

            # 수익률 (round trip fee 반영)
            gross = (exit_px - entry_px) / entry_px
            net = gross - FEE_RT

            trades.append({
                "symbol": sym,
                "ts_entry": ohlcv.iloc[entry_idx]["ts"].isoformat(),
                "ts_exit":  ohlcv.iloc[exit_idx]["ts"].isoformat(),
                "entry": entry_px,
                "exit": exit_px,
                "tp": tp,
                "sl": sl,
                "expiry_h": expiry_h,
                "reason": exit_reason,
                "net": net,
                "entry_mode": st.entry_mode,
            })

    tr_df = pd.DataFrame(trades)
    stats = {"symbol": sym, "trades": len(trades)}
    return tr_df, stats

# ===== 집계 =====
def agg_stats(df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        return pd.Series({"trades": 0, "win_rate": np.nan, "avg_net": np.nan,
                          "median_net": np.nan, "total_net": 0.0})
    wins = (df["net"] > 0).mean() if len(df) else 0.0
    return pd.Series({
        "trades": float(len(df)),
        "win_rate": float(wins),
        "avg_net": float(df["net"].mean()),
        "median_net": float(df["net"].median()),
        "total_net": float(df["net"].sum()),
    })

def load_signals(path: str, group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 표준 컬럼 normalize
    if "ts" not in df.columns:
        raise RuntimeError("signals에 'ts' 컬럼이 없습니다.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "symbol" not in df.columns:
        # 일부 메시지 파싱 필요하면 여기 확장
        raise RuntimeError("signals에 'symbol' 컬럼이 없습니다.")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    # group 필터
    if group == "major":
        df = df[df["symbol"].isin(MAJOR)]
    elif group == "alt":
        df = df[~df["symbol"].isin(MAJOR)]

    # TV 전용 신호만(가능하면 source/host 판단)
    if "source" in df.columns:
        # email/TV 관련만 남기되, 확실하지 않으면 모두 유지
        mask_tv = df["source"].fillna("").str.contains("TV|EMAIL|TRADINGVIEW", case=False, regex=True) \
                  | df["host"].fillna("").str.contains("TRADINGVIEW|EMAIL|PAUL|SUP&RES|ALERT", case=False, regex=True) \
                  | df["message"].fillna("").str.contains("Price in Box|Detected|Level|ALERT", case=False, regex=True)
        # TV가 거의 없을 수 있어 너무 강하게 거르지 않음 — 필요 시 활성화
        # df = df[mask_tv]
    return df.dropna(subset=["ts","symbol"]).reset_index(drop=True)

def build_strategies(expiries: List[int], entry_mode: str) -> List[Strategy]:
    base = [
        ("stable_1.5/1.0", 1.5, 1.0),
        ("aggressive_2.0/1.25", 2.0, 1.25),
        ("scalp_1.0/0.75", 1.0, 0.75),
        ("mid_1.25/1.0", 1.25, 1.0),
        ("mid_1.75/1.25", 1.75, 1.25),
        ("tight_0.8/0.8", 0.8, 0.8),
    ]
    out = []
    for h in expiries:
        for name, tp, sl in base:
            out.append(Strategy(f"{name}_{h}h", tp, sl, h, entry_mode))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals_csv", help="TV 신호 CSV 경로 (예: ./logs/signals_tv.csv)")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--group", choices=["all","major","alt"], default="all")
    ap.add_argument("--expiry", default="4h,8h", help="예: 4h,8h")
    ap.add_argument("--procs", type=int, default=max(2, cpu_count()//2))
    ap.add_argument("--entry-mode", choices=["close","dip"], default="close",
                    help="close=신호봉 다음봉 오픈에서 바로 진입, dip=신호 시점 장중가격 이하로 내려오면 진입")
    args = ap.parse_args()

    df_sig = load_signals(args.signals_csv, args.group)
    if len(df_sig) == 0:
        print("No signals found after filtering.")
        return

    expiries = parse_expiries(args.expiry)
    strategies = build_strategies(expiries, args.entry_mode)

    # 심볼별 작업 분할
    tasks = []
    for sym, rows in df_sig.groupby("symbol"):
        tasks.append({
            "symbol": sym,
            "signals": rows[["ts","symbol"]].copy(),  # 필요한 최소 칼럼만 전달(피클 가볍게)
            "timeframe": args.timeframe,
            "strategies": strategies,
        })

    print(f"[BT] starting with {len(tasks)} symbols using {args.procs} procs... (entry_mode={args.entry_mode})")

    # 멀티프로세싱
    with Pool(processes=args.procs) as pool:
        parts = pool.map(simulate_symbol, tasks)

    trades = pd.concat([p[0] for p in parts if isinstance(p[0], pd.DataFrame) and len(p[0])], ignore_index=True) \
             if parts else pd.DataFrame()

    if len(trades) == 0:
        print("No trades generated.")
        return

    # 집계
    stats = trades.groupby(["entry_mode","expiry_h"], as_index=False).apply(agg_stats)
    # 보기 좋게 정렬
    trades = trades.sort_values(["symbol","ts_entry"]).reset_index(drop=True)

    out_trades = os.path.join(LOGS_DIR, "bt_tv_mp_trades.csv")
    out_stats  = os.path.join(LOGS_DIR, "bt_tv_mp_stats.csv")
    trades.to_csv(out_trades, index=False)
    stats.to_csv(out_stats, index=False)

    print("\n=== TV Backtest (MP) ===")
    print(f"Entry mode : {args.entry_mode}")
    print(f"Trades saved: {out_trades} (rows={len(trades)})")
    print(f"Stats  saved: {out_stats} (rows={len(stats)})\n")
    print(stats)

if __name__ == "__main__":
    # 윈도우 spawn 안전
    main()