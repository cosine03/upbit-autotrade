# -*- coding: utf-8 -*-
import argparse, re
import pandas as pd
import numpy as np

TEXT_COL_CANDIDATES = [
    "event","Event","message","Message","alert_message","Alert Message",
    "notes","Notes","text","Text","description","Description","title","Title"
]

SUPPORT_PAT = re.compile(
    r'(?i)\b('
    r'support|sup|pivot\s*s|s[1-5]\b|'
    r's\s*[-_/ ]?\s*(touch|bounce|break)|'
    r'price\s*in\s*box.*support'
    r')'
)
RESIST_PAT = re.compile(
    r'(?i)\b('
    r'resist|resistance|pivot\s*r|r[1-5]\b|'
    r'r\s*[-_/ ]?\s*(touch|bounce|break)|'
    r'breakout|price\s*in\s*box.*resist'
    r')'
)

def normalize_side(x):
    if not isinstance(x, str): return np.nan
    s = x.strip().lower()
    if s in ("s","sup","support"): return "support"
    if s in ("r","res","resist","resistance"): return "resistance"
    return np.nan

def infer_side_from_text(txt: str) -> str:
    t = (txt or "").lower()
    if SUPPORT_PAT.search(t): return "support"
    if RESIST_PAT.search(t):  return "resistance"
    return "neutral"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="signals csv (tv or enriched)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-text-cols", default="", help="comma-separated extra text cols to scan")
    args = ap.parse_args()

    df = pd.read_csv(args.src)

    # unify timestamp
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    # 1) 기존 side 보존/표준화
    side_std = pd.Series(np.nan, index=df.index)
    if "side" in df.columns:
        side_std = df["side"].map(normalize_side)

    # 2) 텍스트 풀 만들기 (여러 칼럼 결합)
    cols = TEXT_COL_CANDIDATES + [c.strip() for c in args.extra_text_cols.split(",") if c.strip()]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise SystemExit("No text-like columns found. Add --extra-text-cols or ensure one of: " + ", ".join(TEXT_COL_CANDIDATES))
    text = df[cols].astype(str).agg(" | ".join, axis=1)

    # 3) 비어있는 곳만 텍스트로 추론
    need = side_std.isna()
    inferred = text[need].map(infer_side_from_text)
    side_std.loc[need] = inferred

    # 최종 side
    df["side"] = side_std.fillna("neutral")
    df["is_support"] = df["side"].eq("support")
    df["is_resistance"] = df["side"].eq("resistance")

    # touch 숫자(있으면)
    if "event" in df.columns:
        df["touch_n"] = df["event"].astype(str).str.extract(r"(\d+)").iloc[:,0].astype(float)
    else:
        df["touch_n"] = np.nan

    df.to_csv(args.out, index=False, encoding="utf-8")
    print("[LABEL v3] in=", len(df), "| side counts=", df["side"].value_counts().to_dict(), "| scanned cols=", cols)

if __name__ == "__main__":
    main()