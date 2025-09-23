# sr_robust.py
# Upbit/TV 파이프라인 네트워크 안정화 유틸: 요청 타임아웃, 재시도/백오프, OHLCV 정규화

from __future__ import annotations
import time, random
from typing import Optional
import pandas as pd

# --- (1) requests.get 글로벌 타임아웃 몽키패치 ---
DEFAULT_HTTP_TIMEOUT = 8.0

def _monkeypatch_requests_get(timeout_sec: float):
    import requests
    global DEFAULT_HTTP_TIMEOUT
    DEFAULT_HTTP_TIMEOUT = float(timeout_sec) if timeout_sec and timeout_sec > 0 else 8.0
    if getattr(requests.get, "_sr_robust_patched", False):
        return
    _orig_get = requests.get
    def _get_with_default_timeout(url, **kwargs):
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = DEFAULT_HTTP_TIMEOUT
        return _orig_get(url, **kwargs)
    _get_with_default_timeout._sr_robust_patched = True  # type: ignore[attr-defined]
    requests.get = _get_with_default_timeout  # type: ignore[assignment]

def set_default_http_timeout(timeout_sec: float = 8.0):
    """스크립트 시작 직후 1회 호출 권장."""
    _monkeypatch_requests_get(timeout_sec)

# --- (2) 재시도 + 지수 백오프 (+지터) ---
def _sleep_backoff_try(k: int, base_sleep: float):
    time.sleep((base_sleep * (2 ** k)) + random.uniform(0, 0.25))

# --- (3) OHLCV 로더 (sr_engine 우선, pyupbit 대체) + 정규화 ---
def _raw_get_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    # sr_engine.data.get_ohlcv 를 우선 사용
    try:
        from sr_engine.data import get_ohlcv as _get_ohlcv_raw  # type: ignore
        df = _get_ohlcv_raw(symbol, timeframe)
        return df
    except Exception:
        pass

    # fallback: pyupbit
    try:
        import pyupbit as _py
        tf_map = {
            "1m": "minute1", "3m": "minute3", "5m": "minute5", "10m": "minute10",
            "15m": "minute15", "30m": "minute30", "60m": "minute60", "240m": "minute240",
            "1h": "minute60", "4h": "minute240", "8h": "minute240"  # 8h는 240m로 근사(주의)
        }
        tf = tf_map.get(timeframe, timeframe)
        idx_df = _py.get_ohlcv(symbol, interval=tf)
        if idx_df is None or idx_df.empty:
            return None
        df = idx_df.reset_index().rename(columns={
            "index": "ts",
            "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"
        })
        # pyupbit는 보통 tz-naive KST 또는 UTC-naive. 일단 'ts'를 UTC기준 tz-aware로 파싱 후 tz 제거하여 통일
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").tz_localize(None)
        keep = ["ts", "open", "high", "low", "close", "volume"]
        df = df[keep].dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
        return df
    except Exception:
        return None

def get_ohlcv_retry(symbol: str, timeframe: str, retries: int = 4, base_sleep: float = 0.8) -> Optional[pd.DataFrame]:
    """
    네트워크/SSL/일시적 빈 응답 대비 재시도 래퍼.
    성공 시 tz-naive(UTC기준) Datetime64[ns] 로 'ts' 정규화된 DataFrame 반환.
    """
    last_err: Optional[Exception] = None
    for k in range(max(1, retries)):
        try:
            df = _raw_get_ohlcv(symbol, timeframe)
            if df is not None and not df.empty:
                # 최종 안전 정규화
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").tz_localize(None)
                return df.dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
            last_err = RuntimeError("empty dataframe")
        except Exception as e:
            last_err = e
        _sleep_backoff_try(k, base_sleep)
    return None  # 마지막에도 실패시 None

# --- (4) 가벼운 요청 스로틀 (멀티프로세스 과도 동시접속 완화) ---
def micro_throttle(sleep_sec: float):
    if sleep_sec and sleep_sec > 0:
        time.sleep(sleep_sec)