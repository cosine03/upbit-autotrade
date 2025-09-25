# -*- coding: utf-8 -*-
import argparse, re
import pandas as pd
import numpy as np

SUPPORT_PAT = re.compile(r'\b(support|sup|s[1-5]\b|pivot\s*s|s[-_\s]?(touch|bounce|break))', re.I)
RESIST_PAT  = re.compile(r'\b(resist|resistance|r[1-5]\b|pivot\s*r|r[-_\s]?(touch|bounce|break)|breakout)\b', re.I)

def normalize_side(x):
    if not isinstance(x, str):
        return np.nan
    s = x.strip().lower()
    if s in ("s","sup","support"):
        return "support"
    if s in ("r","res","resist","resistance"):
        return "resistance"
    return np.nan

def infer_side_from_event(ev: str) -> str:
    e = (ev or "").lower()
    if SUPPORT_PAT.search(e):
        return "support"
    if RESIST_PAT.search(e):
        return "resistance"
    return "neutral"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="signals csv (tv or enriched)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.src)

    # timestamp 정규화
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    # event 컬럼명 케이스 보정
    evcol = "event" if "event" in df.columns else ("Event" if "Event" in df.columns else None)
    if evcol is None:
        raise SystemExit("input csv must contain an 'event' column")

    # 1) 기존 side 보존 + 표준화
    side_std = pd.Series(np.nan, index=df.index)
    if "side" in df.columns:
        side_std = df["side"].map(normalize_side)

    # 2) 비어있는 곳만 이벤트로 추론
    need_infer = side_std.isna()
    inferred = df.loc[need_infer, evcol].astype(str).map(infer_side_from_event)
    side_std.loc[need_infer] = inferred

    df["side"] = side_std.fillna("neutral")
    df["is_support"] = df["side"].eq("support")
    df["is_resistance"] = df["side"].eq("resistance")

    # optional: touch 숫자 추출
    df["touch_n"] = df[evcol].astype(str).str.extract(r"(\d+)").iloc[:,0].astype(float)

    df.to_csv(args.out, index=False, encoding="utf-8")
    print("[LABEL] in=", len(df), "| side counts=", df["side"].value_counts().to_dict())

if __name__ == "__main__":
    main()