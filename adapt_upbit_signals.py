# -*- coding: utf-8 -*-
import re, pandas as pd, sys
from pathlib import Path

SRC  = sys.argv[1] if len(sys.argv) > 1 else r".\logs\signals_upbit.csv"
OUT1 = sys.argv[2] if len(sys.argv) > 2 else r".\logs\signals_upbit_enriched.csv"
OUT2 = sys.argv[3] if len(sys.argv) > 3 else r".\logs\signals_upbit_enriched_breakout_resist.csv"

df = pd.read_csv(SRC)

# 1) 컬럼 표준화
cols = {c.lower(): c for c in df.columns}
df.columns = [c.lower() for c in df.columns]

# ts → UTC ISO
if "ts" not in df.columns:
    raise SystemExit("signals_upbit.csv 에 'ts' 컬럼이 필요합니다.")
df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

# symbol 보정 (ticker -> symbol)
if "symbol" not in df.columns:
    if "ticker" in df.columns:
        df["symbol"] = df["ticker"]
    else:
        raise SystemExit("symbol/ticker 컬럼이 필요합니다.")

# event 문자열 정규화 (우리가 쓰는 표준 이름으로 매핑)
def norm_event(x:str) -> str:
    s = str(x or "").lower()
    if "box_breakout" in s:   return "box_breakout"
    if "line_breakout" in s:  return "line_breakout"
    if "price_in_box" in s:   return "price_in_box"
    if "level3" in s:         return "level3_detected"
    if "level2" in s:         return "level2_detected"
    if "detected" in s:       return "detected"
    return s or "detected"
df["event"] = df.get("event","detected").astype(str).map(norm_event)

# side 없으면 message에서 추론
def infer_side(row):
    side = str(row.get("side","") or "").strip().lower()
    if side in ("support","resistance"):
        return side
    msg = str(row.get("message","") or "")
    if re.search(r"resistance", msg, re.I): return "resistance"
    if re.search(r"support",    msg, re.I): return "support"
    return ""
df["side"] = df.apply(infer_side, axis=1)

# touches 수치화(있으면)
if "touches" in df.columns:
    df["touches"] = pd.to_numeric(df["touches"], errors="coerce")
else:
    df["touches"] = pd.NA

# distance_pct 없으면 비워둠(필터 자동 skip)
if "distance_pct" not in df.columns:
    df["distance_pct"] = pd.NA

# 표준 컬럼만 보존 + 정렬
keep = ["ts","symbol","event","side","touches","distance_pct","message"]
df = df[[c for c in keep if c in df.columns]].sort_values("ts")

# 저장(전체)
Path(OUT1).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT1, index=False, encoding="utf-8")
print(f"[ADAPT] enriched saved -> {OUT1} (rows={len(df)})")

# 저장(백테스트 기본 조건: resistance & breakout & touches>=3)
sub = df[
    (df["side"]=="resistance") &
    (df["event"].isin(["box_breakout","line_breakout"])) &
    (pd.to_numeric(df["touches"], errors="coerce")>=3)
].copy()

sub.to_csv(OUT2, index=False, encoding="utf-8")
print(f"[ADAPT] breakout+resistance (touches>=3) saved -> {OUT2} (rows={len(sub)})")