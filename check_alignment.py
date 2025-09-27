import pandas as pd, os, sys, glob

SIG = r".\logs\signals_binance_enriched_breakout_resist.csv"  # 네가 돌린 파일 경로로 조정 가능
ROOT = r".\data\ohlcv_binance"                                # 바이낸스 OHLCV 저장 위치
TF   = "15m"

sig = pd.read_csv(SIG)
sig["ts"] = pd.to_datetime(sig["ts"], utc=True, errors="coerce")

rows = []
for sym, g in sig.groupby("symbol"):
    f = os.path.join(ROOT, f"{sym}-{TF}.csv")
    if not os.path.exists(f):
        rows.append((sym, "MISSING_FILE", None, None, None, None, None))
        continue
    o = pd.read_csv(f)
    o["ts"] = pd.to_datetime(o["ts"], utc=True, errors="coerce")
    o = o.sort_values("ts").reset_index(drop=True)
    ots = o["ts"]

    for ts in g["ts"]:
        # 엔트리 바 후보들
        idx_right = int(ots.searchsorted(ts, side="right") - 1)
        idx_left  = int(ots.searchsorted(ts, side="left")  - 1)
        # 최근접 바와 시간차(분)
        nearest_idx = max(min(idx_right, len(ots)-1), 0)
        dt_min = (ts - ots.iloc[nearest_idx]).total_seconds() / 60.0
        rows.append((sym, "OK", ts, ots.iloc[nearest_idx], idx_left, idx_right, round(dt_min,3)))

rep = pd.DataFrame(rows, columns=[
    "symbol","status","ts_sig","ts_bar","idx_left","idx_right","sig_minus_bar_min"
])
print(rep.to_string(index=False))