# estimate_tv_levels.py
# ---------------------
# signals_tv.csv를 읽어 각 신호 시점의 가격과 추정 레벨(박스)을 계산하고
# price와 level 간 괴리율(distance_pct, %)을 추가한 CSV를 저장합니다.
#
# 특징
# - timeframe 컬럼이 비어 있거나 없어도 --timeframe 값으로 자동 보정
# - tz-naive/aware 혼합 안전 처리 (UTC로 통일)
# - 심볼/타임프레임별 로컬 OHLCV CSV 자동 탐색 (여러 경로 시도)
# - 신호 시점 바로 이전 구간(window)에서 레벨 근사:
#     * support  : 최근 구간 'low'의 5% 분위수 (Q5)
#     * resistance: 최근 구간 'high'의 95% 분위수 (Q95)
# - distance_pct = |price - est_level| / est_level * 100
# - 실패 원인(OHLCV 없음, 인덱스 실패 등) 명시 저장
#
# 사용 예:
#   python estimate_tv_levels.py .\logs\signals_tv.csv \
#       --timeframe 15m \
#       --http-timeout 8 --retries 4 --throttle 0.10 \
#       --out .\logs\signals_tv_enriched.csv
#
# OHLCV 파일 규칙(아래 경로들을 순서대로 탐색합니다. 존재하는 첫 파일 사용):
#   ./ohlcv/{symbol}_{timeframe}.csv
#   ./ohlcv/upbit_{symbol}_{timeframe}.csv
#   ./ohlcv/upbit/{symbol}_{timeframe}.csv
# CSV 컬럼은 최소한: ts, open, high, low, close 필요
# ts는 UTC(+00:00) 기준 권장(다른 tz여도 자동 보정됨)
# ---------------------

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


# ---------- 유틸 ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("signals", help="signals_tv.csv (또는 유사 포맷)")
    p.add_argument("--timeframe", default="15m", help="기본 timeframe (CSV에 없거나 비었을 때 보정)")
    p.add_argument("--out", default="./logs/signals_tv_enriched.csv", help="출력 CSV 경로")
    # 아래 3개는 로컬 OHLCV 버전에서는 사용하지 않지만, 인자 수용을 위해 둠(무시)
    p.add_argument("--http-timeout", type=float, default=8.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--throttle", type=float, default=0.10)

    # 레벨 추정 파라미터
    p.add_argument("--lookback-bars", type=int, default=300, help="레벨 추정시 뒤로 볼 바 개수")
    p.add_argument("--q-support", type=float, default=0.05, help="support 레벨 추정 분위수 (0~1)")
    p.add_argument("--q-resistance", type=float, default=0.95, help="resistance 레벨 추정 분위수 (0~1)")
    return p.parse_args()


def log(msg):
    print(msg, flush=True)


def to_utc_ts(s):
    """UTC로 안전 변환 (tz-aware/naive 모두 허용). 실패 시 NaT."""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    out = pd.to_datetime(s, utc=True, errors="coerce")
    return out


def normalize_signals(df, default_tf):
    # 필수 컬럼 보정
    need_cols = ["ts", "event", "side", "symbol", "timeframe"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    # 문자열 정리
    for c in ["event", "side", "symbol", "timeframe"]:
        df[c] = df[c].astype(str).str.strip()

    # timeframe 공백/NaN 보정 -> 기본값
    df.loc[(df["timeframe"] == "") | (df["timeframe"].str.lower() == "nan"), "timeframe"] = default_tf
    df["timeframe"] = df["timeframe"].str.lower()

    # 이벤트/사이드 정규화(소문자)
    df["event"] = df["event"].str.lower()
    df["side"] = df["side"].str.lower()

    # 타임스탬프 UTC 변환
    df["ts"] = to_utc_ts(df["ts"])

    # 심볼 공백 제거
    df["symbol"] = df["symbol"].str.replace(r"\s+", "", regex=True)

    # 유효행만 남기기
    keep = df["ts"].notna() & df["symbol"].ne("") & df["timeframe"].ne("") & df["event"].ne("") & df["side"].ne("")
    return df.loc[keep].reset_index(drop=True)


def find_ohlcv_path(symbol, timeframe):
    """여러 경로 후보 중 존재하는 CSV 첫번째 반환, 없으면 None"""
    candidates = [
        f"./ohlcv/{symbol}_{timeframe}.csv",
        f"./ohlcv/upbit_{symbol}_{timeframe}.csv",
        f"./ohlcv/upbit/{symbol}_{timeframe}.csv",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_ohlcv(symbol, timeframe):
    """로컬 OHLCV 로드(ts, open, high, low, close). UTC로 정규화."""
    path = find_ohlcv_path(symbol, timeframe)
    if path is None:
        return None, f"OHLCV file not found for {symbol} {timeframe}"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"read_csv error: {e}"

    # 필수 컬럼 체크
    for c in ["ts", "open", "high", "low", "close"]:
        if c not in df.columns:
            return None, f"missing column '{c}' in {path}"

    # 시간 정규화
    df["ts"] = to_utc_ts(df["ts"])
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df, None


def search_index_at_or_before(ts_array, key_ts):
    """
    ts_array: 정렬된 UTC 타임스탬프 시리즈/배열
    key_ts: UTC Timestamp
    반환: key_ts가 속한(또는 바로 이전) 바의 인덱스 (없으면 -1)
    """
    # numpy datetime64[ns]로 통일
    a = pd.Series(ts_array).astype("datetime64[ns]").to_numpy()
    k = pd.to_datetime(key_ts, utc=True)
    if pd.isna(k):
        return -1
    k64 = np.datetime64(k.to_datetime64())
    # 오른쪽 삽입 후 -1: '이전' 위치
    idx = int(np.searchsorted(a, k64, side="right")) - 1
    return idx


def estimate_level(window_df, side, q_support=0.05, q_resistance=0.95):
    """
    간단 근사:
      support    -> window low의 q_support 분위수
      resistance -> window high의 q_resistance 분위수
    """
    if window_df.empty:
        return np.nan, "empty window"

    if side == "support":
        level = float(np.nanquantile(window_df["low"].to_numpy(), q_support))
        return level, ""
    elif side == "resistance":
        level = float(np.nanquantile(window_df["high"].to_numpy(), q_resistance))
        return level, ""
    else:
        return np.nan, f"unknown side '{side}'"


# ---------- 메인 파이프라인 ----------

def main():
    args = parse_args()

    # 입력 로드
    try:
        df_sig_raw = pd.read_csv(args.signals)
    except Exception as e:
        log(f"[EST] ERROR: cannot read signals file: {e}")
        sys.exit(1)

    df_sig = normalize_signals(df_sig_raw.copy(), default_tf=args.timeframe)

    # Paul 지표 이벤트만 유지(혹시 다른 이벤트가 섞였을 경우)
    valid_events = {"level2_detected", "level3_detected", "price_in_box", "box_breakout", "line_breakout"}
    df_sig = df_sig[df_sig["event"].isin(valid_events)].reset_index(drop=True)

    if df_sig.empty:
        # 아무 행도 없으면 그대로 저장만
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        df_sig.to_csv(args.out, index=False)
        log(f"[EST] done. saved -> {args.out}")
        log(f"[EST] success 0/0 rows")
        return

    # 결과 컬럼 준비
    df_sig["price_at_signal"] = np.nan
    df_sig["est_level"] = np.nan
    df_sig["distance_pct"] = np.nan
    df_sig["bars_lookback_used"] = np.nan
    df_sig["est_note"] = ""

    # 심볼/타임프레임 단위로 처리
    total = len(df_sig)
    ok = 0

    # 미리 심볼/TF별 OHLCV 로드 캐시
    cache = {}

    # 정렬(안전)
    df_sig = df_sig.sort_values(["symbol", "timeframe", "ts"]).reset_index(drop=True)

    for i, row in df_sig.iterrows():
        sym = row["symbol"]
        tf = row["timeframe"]
        ts = row["ts"]
        side = row["side"]

        key = (sym, tf)
        if key not in cache:
            ohlcv, err = load_ohlcv(sym, tf)
            cache[key] = (ohlcv, err)
        else:
            ohlcv, err = cache[key]

        if ohlcv is None:
            df_sig.at[i, "est_note"] = f"ohlcv_error: {err}"
            continue

        # 시그널 시점 이전 바 인덱스
        idx = search_index_at_or_before(ohlcv["ts"], ts)
        if idx < 0:
            df_sig.at[i, "est_note"] = "index_not_found_before_ts"
            continue

        # 신호 시점 가격(해당 바 close로 정의)
        price = float(ohlcv.at[idx, "close"])
        df_sig.at[i, "price_at_signal"] = price

        # lookback window
        i0 = max(0, idx - int(args.lookback_bars) + 1)
        window = ohlcv.iloc[i0:idx + 1].copy()

        # 레벨 근사
        level, note = estimate_level(
            window, side,
            q_support=float(args.q_support),
            q_resistance=float(args.q_resistance),
        )
        if not np.isfinite(level) or level <= 0:
            df_sig.at[i, "est_note"] = note if note else "invalid_level"
            continue

        df_sig.at[i, "est_level"] = level
        df_sig.at[i, "bars_lookback_used"] = len(window)

        # 괴리율(%)
        dist_pct = abs(price - level) / level * 100.0
        df_sig.at[i, "distance_pct"] = dist_pct

        ok += 1
        # 가벼운 진행 표시(큰 파일 대비 과도한 출력 방지)
        if ok % 100 == 0:
            log(f"[EST] progress: {ok}/{total} ok")

    # 저장
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    df_sig.to_csv(args.out, index=False)

    log(f"[EST] done. saved -> {args.out}")
    log(f"[EST] success {ok}/{total} rows")


if __name__ == "__main__":
    main()