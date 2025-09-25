# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd, os

def auto_scale_and_mask(x: pd.Series, thr: float):
    v = pd.to_numeric(x, errors="coerce").values
    scale = "ratio" if (np.nanmax(v) <= 1.0) else "percent"
    if scale == "percent":
        v = v / 100.0
    return (v <= thr), scale

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="labeled signals csv (has 'side')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-max", type=float, default=0.00025, help="ratio (0.00025=0.025%)")
    ap.add_argument("--include-substr", default="support", help="comma-separated substrings; empty to disable")
    ap.add_argument("--require-same-level", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = pd.read_csv(args.src)
    if "side" not in df.columns:
        raise SystemExit("input must have 'side' column (run labeler first)")
    if "event" not in df.columns and "Event" in df.columns:
        df = df.rename(columns={"Event":"event"})

    # ts 정규화
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    sub = df[df["side"].eq("support")].copy()

    # 부분문자열 필터 (빈 문자열이면 비활성화)
    keys = [k.strip().lower() for k in (args.include_substr or "").split(",") if k.strip()]
    if keys:
        sub = sub[sub["event"].astype(str).str.lower().map(lambda s: any(k in s for k in keys))]

    # distance_pct 필터(있을 때만)
    if "distance_pct" in sub.columns:
        mask, scale = auto_scale_and_mask(sub["distance_pct"], args.dist_max)
        before = len(sub); sub = sub[mask].copy()
        print(f"[FILTER] distance_pct ({scale}) <= {args.dist_max}: {before}->{len(sub)}")
    else:
        print("[FILTER] distance_pct missing: skip")

    # 동일 레벨 강제 (레벨류 칼럼이 있을 때만)
    if args.require-same-level:
        level_cols = [c for c in sub.columns if "level" in c.lower()]
        if level_cols:
            before = len(sub)
            sub = sub[~sub[level_cols].isna().all(axis=1)].copy()
            print(f"[FILTER] require_same_level: {before}->{len(sub)}")
        else:
            print("[FILTER] require_same_level: no level-like columns; skip")

    # 0행이어도 헤더 유지
    if sub.empty:
        sub = sub.head(0)
    sub.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OUT] saved -> {args.out} rows={len(sub)}")

if __name__ == "__main__":
    main()