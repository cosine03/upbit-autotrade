# -*- coding: utf-8 -*-
"""
TV 신호 기반 백테스트 (폴 지표 / 종가 엔트리 / 만료 4h&8h / 롱온리 / 수수료 0.1% 왕복)

- 입력: signals_tv.csv (칼럼 예시: ts,event,side,level,touches,symbol,timeframe,extra,source,host,message)
- 엔트리: 신호 발생 봉의 "다음 봉 종가"로 진입 (close-entry)
- 만료: 4시간 / 8시간 (둘 다 수행)
- TP/SL: 기본 1.5% / 1.0% (옵션으로 변경 가능)
- 수수료: 왕복 0.1% (진입 0.05% + 청산 0.05%) 반영
- touches 필터: >=2 와 >=3 두 가지 시나리오를 각각 수행 (touches 컬럼 또는 메시지에서 추출)

실행 예:
  python backtest_tv_breakout_close_4h8h.py ./logs/signals_tv.csv --timeframe 15m --tp 1.5 --sl 1.0

출력:
  ./logs/bt_tv_breakout_4h8h_trades_t2.csv
  ./logs/bt_tv_breakout_4h8h_stats_t2.csv
  ./logs/bt_tv_breakout_4h8h_trades_t3.csv
  ./logs/bt_tv_breakout_4h8h_stats_t3.csv
"""

import os
import re
import argparse
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

# upbit OHLCV 로더 (sr_engine.data 이용) — 없으면 pyupbit 직접 사용
try:
    from sr_engine.data import get_ohlcv
except Exception:
    get_ohlcv = None
try:
    import pyupbit
except Exception:
    pyupbit = None


# -------------------- 유틸: 시간/타임프레임 --------------------
def tf_to_minutes(tf: str) -> int:
    s = tf.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("d"):
        return int(s[:-1]) * 60 * 24
    return 15


def to_utc_ts(s) -> pd.Timestamp:
    """입력 문자열/타임스탬프를 UTC aware Timestamp로 표준화."""
    if isinstance(s, pd.Timestamp):
        if s.tzinfo is None:
            return s.tz_localize("UTC")
        return s.tz_convert("UTC")
    # 문자열 포함 모든 입력을 안전하게 파싱
    return pd.to_datetime(s, utc=True, errors="coerce")


def series_to_ns_utc(s: pd.Series) -> np.ndarray:
    """
    OHLCV의 ts Series를 'UTC 기준 tz-naive datetime64[ns]' numpy 배열로 통일.
    tz-aware면 UTC로 맞춘 뒤 tz를 제거하고, tz-naive면 UTC로 가정해 ns로 변환.
    """
    # s를 반드시 pandas Series datetime으로
    s = pd.to_datetime(s, errors="coerce", utc=False)

    # tz-aware 인지 판별
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        # UTC로 맞추고 tz 제거 → tz-naive
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        # 이미 tz-naive라면 UTC로 간주(Upbit/TV 모두 UTC 타임스탬프 기반)
        # 혹시 섞여있을 수 있으니 한번 더 보정
        s = s.dt.tz_localize(None)

    # 이제 tz-naive 상태이므로 안전하게 ns로 변환
    return s.astype("datetime64[ns]").to_numpy()


def idx_of_bar(ts64: np.ndarray, key_ts: pd.Timestamp) -> int:
    """
    ts64 : datetime64[ns] numpy (OHLCV 타임스탬프)
    key_ts: UTC-aware pandas.Timestamp

    반환: key_ts가 속한(또는 직전) 바의 인덱스.
    """
    k64 = np.datetime64(to_utc_ts(key_ts))
    idx = int(np.searchsorted(ts64, k64, side="right")) - 1
    return max(0, min(idx, len(ts64) - 1))


# -------------------- 신호 로드/정리 --------------------
TOUCH_HINT = re.compile(r"(touch(?:es)?\s*[:=]\s*(\d+))|(Min\s*Touches\s*:\s*(\d+))", re.IGNORECASE)

def infer_touches(row: pd.Series) -> int:
    # 우선 'touches' 칼럼 신뢰
    if "touches" in row and pd.notna(row["touches"]):
        try:
            return int(row["touches"])
        except Exception:
            pass
    # 메시지에서 추출
    msg = str(row.get("message", "") or "")
    m = TOUCH_HINT.search(msg)
    if m:
        for g in (m.group(2), m.group(4)):
            if g:
                try:
                    return int(g)
                except Exception:
                    pass
    # level2/level3 패턴 보정
    ev = str(row.get("event", "") or "").lower()
    if "level3" in ev:
        return 3
    if "level2" in ev:
        return 2
    # 기본값
    return 2


def load_signals_tv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 필수열 보정
    for c in ("ts", "symbol"):
        if c not in df.columns:
            raise ValueError(f"signals file missing column: {c}")
    # UTC aware
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "symbol"]).copy()

    # side 기본값 보정 (없거나 공백이면 resistance로 가정: 롱 전용)
    if "side" not in df.columns:
        df["side"] = "resistance"
    df["side"] = df["side"].fillna("resistance").str.lower()

    # touches 보강
    if "touches" not in df.columns:
        df["touches"] = None
    df["touches"] = df.apply(infer_touches, axis=1)

    # 심볼 정리
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    # TV 특성상 'KRW-' 접두가 있어야 pyupbit/get_ohlcv가 이해함
    df = df[df["symbol"].str.startswith("KRW-")].copy()

    # 시간순
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df


# -------------------- OHLCV 로딩 --------------------
def fetch_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """sr_engine.data.get_ohlcv 우선, 실패 시 pyupbit 백업."""
    iv = timeframe
    try:
        if get_ohlcv is not None:
            df = get_ohlcv(symbol, iv)
            if df is not None and not df.empty:
                # 표준화
                if "ts" not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex):
                        ts = df.index
                        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
                        df = df.reset_index().rename(columns={"index": "ts"})
                        df["ts"] = ts
                else:
                    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                df = df.dropna(subset=["ts"]).reset_index(drop=True)
                return df[["ts","open","high","low","close","volume"]].copy()
    except Exception:
        pass

    if pyupbit is None:
        return None

    # pyupbit 백업
    try:
        df = pyupbit.get_ohlcv(ticker=symbol, interval=iv, count=4000)
        if df is None or df.empty:
            return None
        ts = df.index
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        df = df.reset_index().rename(columns={"index": "ts"})
        df["ts"] = ts
        return df[["ts","open","high","low","close","volume"]].copy()
    except Exception:
        return None


# -------------------- 시뮬 엔진 --------------------
def simulate_symbol(
    symbol: str,
    ohlcv: pd.DataFrame,
    sig_rows: pd.DataFrame,
    timeframe: str,
    tp_pct: float,
    sl_pct: float,
    fee_rt: float,
    expiry_h: float
) -> pd.DataFrame:
    """
    - entry: 신호봉의 '다음 봉 종가' (close-entry)
    - 만료: expiry_h 시간 뒤 바 오픈 전까지
    - TP/SL: intrabar high/low로 체결 가정 (동시 충족 시 보수적으로 SL 우선 처리)
    - 수수료: 왕복 fee_rt 반영
    """
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    # UTC 보정 + numpy 변환
    ohlcv = ohlcv.copy()
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True, errors="coerce")
    ohlcv = ohlcv.dropna(subset=["ts"]).reset_index(drop=True)
    ts64 = series_to_ns_utc(ohlcv["ts"])
    tf_min = tf_to_minutes(timeframe)
    bars_expiry = int(np.ceil((expiry_h * 60) / max(1, tf_min)))

    out = []

    for _, s in sig_rows.iterrows():
    sig_ts = pd.to_datetime(s["ts"], utc=True, errors="coerce")

    # tz-aware → UTC로 맞추고 tz 제거
    if sig_ts.tzinfo is not None:
        sig_ts = sig_ts.tz_convert("UTC").tz_localize(None)
    else:
        sig_ts = sig_ts.tz_localize(None)

    # numpy datetime64[ns] 로 변환
    sig_ts64 = np.datetime64(sig_ts, "ns")
        i_ent = i_sig + 1                     # 다음 바 종가로 진입
        if i_ent >= len(ohlcv):
            continue

        entry_ts = ohlcv["ts"].iloc[i_ent]
        entry_px = float(ohlcv["close"].iloc[i_ent])

        # 만료 인덱스: entry 이후 bars_expiry 바까지만 관찰
        i_end = min(len(ohlcv)-1, i_ent + bars_expiry)

        # TP/SL 기준가
        tp_price = entry_px * (1.0 + tp_pct/100.0)
        sl_price = entry_px * (1.0 - sl_pct/100.0)

        # 수수료: 왕복 fee_rt (0.001 = 0.1%)
        fee = fee_rt

        hit_idx = None
        exit_price = None
        exit_reason = None

        # 엔트리 다음 바부터 순회
        for j in range(i_ent+1, i_end+1):
            hi = float(ohlcv["high"].iloc[j])
            lo = float(ohlcv["low"].iloc[j])

            # 동시 충족 보수적으로 SL 우선
            if lo <= sl_price:
                hit_idx = j
                exit_price = sl_price
                exit_reason = "SL"
                break
            if hi >= tp_price:
                hit_idx = j
                exit_price = tp_price
                exit_reason = "TP"
                break

        # TP/SL 미충족 → 만료 종가 청산
        if hit_idx is None:
            hit_idx = i_end
            exit_price = float(ohlcv["close"].iloc[hit_idx])
            exit_reason = "EXP"

        # 순손익(%) 계산: 왕복 수수료 차감
        gross = (exit_price - entry_px) / entry_px
        net = gross - fee

        out.append({
            "symbol": symbol,
            "signal_ts": sig_ts.isoformat(),
            "entry_ts": entry_ts.isoformat(),
            "exit_ts": ohlcv["ts"].iloc[hit_idx].isoformat(),
            "entry": entry_px,
            "exit": exit_price,
            "net_pct": net,
            "reason": exit_reason,
        })

    return pd.DataFrame(out)


def run_strategy(
    df_sig: pd.DataFrame,
    timeframe: str = "15m",
    tp: float = 1.5,
    sl: float = 1.0,
    fee_rt: float = 0.001,
    min_touches: int = 2,
    expiry_hours_list: Tuple[float, ...] = (4.0, 8.0),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    touches ≥ min_touches에 대해 expiry 후보(4h,8h)를 각각 시뮬.
    """
    # 대상 심볼
    symbols = sorted(df_sig["symbol"].unique().tolist())
    trades_all = []

    for sym in symbols:
        # 신호 필터: resistance + touches ≥ min_touches
        rows = df_sig[(df_sig["symbol"] == sym) & (df_sig["side"].str.lower() == "resistance")].copy()
        rows = rows[rows["touches"].astype(int) >= int(min_touches)].copy()
        if rows.empty:
            continue

        # OHLCV 로드
        ohlcv = fetch_ohlcv(sym, timeframe)
        if ohlcv is None or ohlcv.empty:
            print(f"[{sym}] OHLCV load failed (skip).")
            continue

        for eh in expiry_hours_list:
            tr = simulate_symbol(
                sym, ohlcv, rows, timeframe, tp, sl, fee_rt, eh
            )
            if not tr.empty:
                tr["expiry_h"] = eh
                tr["strategy"] = f"touch{min_touches}_{tp}/{sl}_{int(eh)}h"
                trades_all.append(tr)

    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame(
        columns=["symbol","signal_ts","entry_ts","exit_ts","entry","exit","net_pct","reason","expiry_h","strategy"]
    )

    # 통계
    def agg(g: pd.DataFrame) -> pd.Series:
        if g.empty:
            return pd.Series({"trades": 0, "win_rate": 0.0, "avg_net": 0.0, "median_net": 0.0, "total_net": 0.0})
        wins = (g["net_pct"] > 0).mean() if len(g) else 0.0
        return pd.Series({
            "trades": float(len(g)),
            "win_rate": float(wins),
            "avg_net": float(g["net_pct"].mean()),
            "median_net": float(g["net_pct"].median()),
            "total_net": float(g["net_pct"].sum()),
        })

    stats = trades.groupby("strategy", as_index=False, dropna=False).apply(agg)
    return trades, stats


# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals", type=str, help="signals_tv.csv 경로")
    ap.add_argument("--timeframe", type=str, default="15m", help="예: 15m, 5m, 1h ...")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001, help="왕복 수수료율 (기본 0.001=0.1%)")
    args = ap.parse_args()

    df_sig = load_signals_tv(args.signals)
    print(f"[BT] signals rows={len(df_sig)}, symbols={df_sig['symbol'].nunique()}, timeframe={args.timeframe}")

    os.makedirs("./logs", exist_ok=True)

    # === touches ≥ 2 ===
    t2_trades, t2_stats = run_strategy(
        df_sig,
        timeframe=args.timeframe,
        tp=args.tp,
        sl=args.sl,
        fee_rt=args.fee,
        min_touches=2,
        expiry_hours_list=(4.0, 8.0),
    )
    print("=== touches≥2 ===")
    if not t2_stats.empty:
        print(t2_stats.sort_values("strategy"))
    else:
        print("(no trades)")

    t2_trades_path = "./logs/bt_tv_breakout_4h8h_trades_t2.csv"
    t2_stats_path  = "./logs/bt_tv_breakout_4h8h_stats_t2.csv"
    t2_trades.to_csv(t2_trades_path, index=False)
    t2_stats.to_csv(t2_stats_path, index=False)

    # === touches ≥ 3 ===
    t3_trades, t3_stats = run_strategy(
        df_sig,
        timeframe=args.timeframe,
        tp=args.tp,
        sl=args.sl,
        fee_rt=args.fee,
        min_touches=3,
        expiry_hours_list=(4.0, 8.0),
    )
    print("=== touches≥3 ===")
    if not t3_stats.empty:
        print(t3_stats.sort_values("strategy"))
    else:
        print("(no trades)")

    t3_trades_path = "./logs/bt_tv_breakout_4h8h_trades_t3.csv"
    t3_stats_path  = "./logs/bt_tv_breakout_4h8h_stats_t3.csv"
    t3_trades.to_csv(t3_trades_path, index=False)
    t3_stats.to_csv(t3_stats_path, index=False)

    print("\n[BT] 저장 완료.")
    print(f"  → {t2_trades_path}")
    print(f"  → {t2_stats_path}")
    print(f"  → {t3_trades_path}")
    print(f"  → {t3_stats_path}")


if __name__ == "__main__":
    main()