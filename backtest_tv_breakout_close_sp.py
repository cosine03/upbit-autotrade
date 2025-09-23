# -*- coding: utf-8 -*-
"""
TV signals 기반 싱글프로세스 백테스트 (Breakout on Close, Long-only, 24h 만기)

- 입력: signals_tv.csv (열: ts,event,side,level,touches,symbol,timeframe,extra,source,host,message)
- 엔트리 규칙(저항/롱 기준):
    1) TV 신호가 'resistance'이고, touches >= T (T=2 또는 3) 인 "준비" 신호를 찾는다.
    2) 신호가 발생한 봉(i_sig)의 다음 봉부터, '종가가 신호봉 종가보다 위에서 마감'하는 첫 봉 j를 찾는다.
    3) j+1 봉의 시가에 진입한다. (브레이크아웃 종가 확인 후 다음 봉 오픈 체결)
- 청산:
    - TP/SL 퍼센트(예: TP=1.5%, SL=1.0%), 라운드트립 수수료 0.1% 반영
    - 24시간(= 96개 캔들 @15m) 만기 시 해당 시점의 종가로 청산
    - intra-bar 충돌 시 SL 우선(보수적 가정)
- 출력:
    - ./logs/bt_tv_breakout_trades.csv
    - ./logs/bt_tv_breakout_stats.csv
"""

import os
import sys
import time
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# pyupbit 이용 (sr_engine.data.get_ohlcv 와 동일 인터페이스로 맞춤)
try:
    import pyupbit
except Exception:
    pyupbit = None

LOGS_DIR = "./logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ---- 유틸: 타임프레임 → 분 ----
def tf_minutes(tf: str) -> int:
    s = tf.lower().strip()
    if s.endswith("m"): return int(s[:-1])
    if s.endswith("h"): return int(s[:-1]) * 60
    if s.endswith("d"): return int(s[:-1]) * 60 * 24
    return 15

# ---- OHLCV 로딩/캐시 (싱글프로세스) ----
def get_ohlcv(symbol: str, timeframe: str="15m", count: int=1000) -> pd.DataFrame:
    """
    pyupbit.get_ohlcv 를 감싸 UTC ts 컬럼을 보장.
    count=200*5=1000 정도로 최근 데이터 확보 (필요시 늘리세요).
    """
    if pyupbit is None:
        raise RuntimeError("pyupbit not installed. `pip install pyupbit` 필요")

    iv = {
        "1m":"minute1","3m":"minute3","5m":"minute5","10m":"minute10","15m":"minute15","30m":"minute30",
        "60m":"minute60","240m":"minute240","4h":"minute240","1d":"day"
    }.get(timeframe, "minute15")

    # pyupbit는 한 번에 최대 200개 → 여러 번 쌓아서 1000개 수준 확보
    need = max(200, count)
    out: List[pd.DataFrame] = []
    to = None
    remain = need
    for _ in range(6):  # 200 * 6 = 1200
        sz = min(200, remain)
        df = pyupbit.get_ohlcv(ticker=symbol, interval=iv, count=sz, to=to)
        if df is None or df.empty:
            break
        df = df.sort_index()
        to = df.index[0]  # 이전 구간 anchor
        out.append(df)
        remain -= len(df)
        if remain <= 0:
            break

    if not out:
        raise RuntimeError(f"OHLCV load failed for {symbol}")

    df = pd.concat(out, axis=0).drop_duplicates().sort_index()
    df = df.reset_index().rename(columns={"index":"ts"})
    # tz 보정 → UTC
    if isinstance(df["ts"].dtype, pd.DatetimeTZDtype):
        df["ts"] = df["ts"].dt.tz_convert("UTC")
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 열 이름 표준화
    cols = ["ts","open","high","low","close","volume","value"]
    existing = [c for c in cols if c in df.columns]
    # 일부 pyupbit 버전은 value가 없을 수 있음 → 보정
    if "value" not in df.columns:
        df["value"] = np.nan
    df = df[["ts","open","high","low","close","volume","value"]]
    return df.reset_index(drop=True)

# ---- 인덱스 탐색: "신호시간이 포함된 봉"과 "다음 봉" ----
def idx_of_bar(ts: np.ndarray, when_utc: pd.Timestamp) -> int:
    """
    ts: tz-aware pandas Timestamps 배열(np.ndarray)
    when_utc: tz-aware UTC Timestamp
    반환: when_utc가 포함된 봉의 인덱스(i_sig), 없으면 -1
    """
    # np.searchsorted 쓸 때는 np.datetime64(UTC)로 맞춰준다
    key = when_utc.to_datetime64()
    # 'right'로 넣고 -1 하면 포함된 바 (<= key)의 마지막 인덱스
    idx = int(np.searchsorted(ts, key, side="right")) - 1
    if idx < 0 or idx >= len(ts):
        return -1
    return idx

def first_breakout_close_index(closes: np.ndarray, start_idx: int, ref_close: float) -> int:
    """
    start_idx 부터 '종가가 ref_close를 상향 돌파' 하는 첫 봉 인덱스 반환.
    없으면 -1
    """
    for j in range(max(start_idx, 0), len(closes)):
        if closes[j] > ref_close:
            return j
    return -1

# ---- 트레이드 시뮬 ----
def simulate_symbol_breakout_close(
    symbol: str,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float = 0.001,   # 0.1% 라운드트립
    timeframe: str = "15m",
    expiry_hours: float = 24.0,
) -> List[Dict]:
    """
    - signals: 해당 심볼의 TV 'resistance' 후보 신호들(미리 touches 필터링된 상태 권장)
    - 엔트리: 브레이크아웃 종가 확인(j), j+1 봉 시가 진입
    - TP/SL/만기: intrabar 충돌 시 SL 우선(보수적)
    """
    if ohlcv is None or ohlcv.empty or signals.empty:
        return []

    ts = ohlcv["ts"].to_numpy()  # tz-aware pandas Timestamp np.ndarray
    open_ = ohlcv["open"].to_numpy(dtype=float)
    high_ = ohlcv["high"].to_numpy(dtype=float)
    low__ = ohlcv["low"].to_numpy(dtype=float)
    close = ohlcv["close"].to_numpy(dtype=float)

    tf_min = tf_minutes(timeframe)
    bars_per_24h = int(round(24 * 60 / tf_min))
    bars_expiry = int(round(expiry_hours * 60 / tf_min))

    trades: List[Dict] = []

    for _, s in signals.iterrows():
        # 신호 시간 tz 보정
        sig_ts = pd.Timestamp(s["ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.tz_localize("UTC")
        else:
            sig_ts = sig_ts.tz_convert("UTC")

        i_sig = idx_of_bar(ts, sig_ts)
        if i_sig < 0:
            continue

        # "신호봉 종가"를 레퍼런스로 사용(레벨 상단 근사치로 사용)
        ref_close = float(close[i_sig])

        # 브레이크아웃 종가 찾기 (신호 다음봉부터 검색)
        j = first_breakout_close_index(close, i_sig + 1, ref_close)
        if j < 0:
            continue

        # 엔트리는 j+1 봉 오픈
        i_entry = j + 1
        if i_entry >= len(ts):
            continue

        entry_ts = ts[i_entry]
        entry_px = float(open_[i_entry])

        # TP/SL 가격
        tp_px = entry_px * (1.0 + tp_pct / 100.0)
        sl_px = entry_px * (1.0 - sl_pct / 100.0)

        # 만기 인덱스
        i_exp = min(i_entry + bars_expiry, len(ts) - 1)

        exit_px = None
        exit_ts = None
        exit_reason = None

        # i_entry+1 부터 체킹 (엔트리 봉 내 체결 X 가정)
        k_start = i_entry + 1
        for k in range(k_start, i_exp + 1):
            # SL 우선
            if float(low__[k]) <= sl_px:
                exit_px = sl_px
                exit_ts = ts[k]
                exit_reason = "SL"
                break
            # TP
            if float(high_[k]) >= tp_px:
                exit_px = tp_px
                exit_ts = ts[k]
                exit_reason = "TP"
                break

        if exit_px is None:
            exit_px = float(close[i_exp])
            exit_ts = ts[i_exp]
            exit_reason = "EXPIRY"

        # 수수료(왕복) 반영한 순익률
        gross = (exit_px - entry_px) / entry_px
        net = gross - fee_rt

        trades.append({
            "symbol": symbol,
            "signal_ts": sig_ts.isoformat(),
            "signal_ref_close": ref_close,
            "entry_ts": pd.Timestamp(entry_ts).isoformat(),
            "entry_px": entry_px,
            "exit_ts": pd.Timestamp(exit_ts).isoformat(),
            "exit_px": exit_px,
            "reason": exit_reason,
            "tp": tp_pct,
            "sl": sl_pct,
            "expiry_h": expiry_hours,
            "net_ret": net,
        })

    return trades

# ---- 통계 ----
def summarize(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["trades","win_rate","avg_net","median_net","total_net"])
    wins = (trades_df["net_ret"] > 0)
    out = pd.DataFrame({
        "trades": [len(trades_df)],
        "win_rate": [wins.mean()],
        "avg_net": [trades_df["net_ret"].mean()],
        "median_net": [trades_df["net_ret"].median()],
        "total_net": [trades_df["net_ret"].sum()],
    })
    return out

# ---- 메인 ----
def main():
    ap = argparse.ArgumentParser(description="Single-process TV breakout-on-close backtest")
    ap.add_argument("signals", help="path to signals_tv.csv")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--expiry", default="24h", help="예: 24h")
    ap.add_argument("--tp", type=float, default=1.5, help="take profit %")
    ap.add_argument("--sl", type=float, default=1.0, help="stop loss %")
    ap.add_argument("--fee", type=float, default=0.001, help="roundtrip fee (e.g., 0.001 = 0.1%)")
    ap.add_argument("--touches", type=int, default=2, help="필터: touches >= 이 값 (2 또는 3 등)")
    ap.add_argument("--touches_alt", type=int, default=3, help="비교용: touches >= 이 값(두 번째 시나리오)")
    args = ap.parse_args()

    expiry_hours = 24.0
    if args.expiry.endswith("h"):
        expiry_hours = float(args.expiry[:-1])

    # 1) 신호 로드 & 정규화
    df = pd.read_csv(args.signals)
    # 컬럼 보정
    need_cols = ["ts","event","side","level","touches","symbol","timeframe"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = ""
    # ts → UTC
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    # 표준화
    df["side"] = df["side"].astype(str).str.lower()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timeframe"] = df["timeframe"].replace("", "15m").fillna("15m")
    df["touches"] = pd.to_numeric(df["touches"], errors="coerce").fillna(0).astype(int)

    # (TV 로그 특성 상 event 값이 다양할 수 있음) 'resistance' 신호만 사용
    df = df[df["side"] == "resistance"].copy()
    if df.empty:
        print("No resistance signals.")
        return

    # 2) 심볼 목록
    symbols = sorted(df["symbol"].unique().tolist())
    print(f"[BT] symbols={len(symbols)} signals(rows)={len(df)} timeframe={args.timeframe}")

    all_trades = []
    # 두 가지 터치 기준(시나리오 A/B) 비교 → 동일 루프에서 각각 결과 생성
    for touches_min in [args.touches, args.touches_alt]:
        print(f"[BT] scenario: touches>={touches_min}, TP={args.tp} SL={args.sl}, expiry={expiry_hours}h")
        trades_scn = []
        for sym in symbols:
            rows = df[(df["symbol"] == sym) & (df["touches"] >= touches_min)]
            if rows.empty:
                continue
            # OHLCV
            try:
                ohlcv = get_ohlcv(sym, timeframe=args.timeframe, count=1000)
            except Exception as ex:
                print(f"[{sym}] OHLCV load error: {ex!r}")
                continue

            tr = simulate_symbol_breakout_close(
                sym, ohlcv, rows, tp_pct=args.tp, sl_pct=args.sl,
                fee_rt=args.fee, timeframe=args.timeframe, expiry_hours=expiry_hours
            )
            for r in tr:
                r["scenario"] = f"touches>={touches_min}"
            trades_scn.extend(tr)

        if not trades_scn:
            print(f"[BT] scenario touches>={touches_min}: no trades")
        all_trades.extend(trades_scn)

    # 3) 저장/요약
    trades_df = pd.DataFrame(all_trades)
    out_trades = os.path.join(LOGS_DIR, "bt_tv_breakout_trades.csv")
    out_stats  = os.path.join(LOGS_DIR, "bt_tv_breakout_stats.csv")

    if trades_df.empty:
        trades_df.to_csv(out_trades, index=False)
        pd.DataFrame(columns=["scenario","trades","win_rate","avg_net","median_net","total_net"]).to_csv(out_stats, index=False)
        print("No trades generated.")
        return

    stats = (trades_df
             .groupby(["scenario"], as_index=False)
             .apply(lambda g: pd.concat([pd.DataFrame({"scenario":[g.name]}), summarize(g.reset_index(drop=True))], axis=1))
             .reset_index(drop=True))

    trades_df.to_csv(out_trades, index=False)
    stats.to_csv(out_stats, index=False)

    print(f"\n=== TV Breakout-on-Close Backtest (Single Proc) ===")
    print(f"Trades saved: {out_trades} (rows={len(trades_df)})")
    print(f"Stats  saved: {out_stats} (rows={len(stats)})\n")
    print(stats.to_string(index=False))
    print()

if __name__ == "__main__":
    main()