# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd

def auto_scale_and_mask(x: pd.Series, thr: float):
    """distance_pct 스케일 자동감지, thr(비율) 기준 마스크 반환"""
    v = pd.to_numeric(x, errors="coerce").values
    scale = "ratio" if (np.nanmax(v) <= 1.0) else "percent"
    if scale == "percent":
        v = v / 100.0
    mask = v <= thr
    return mask, scale

def same_level_mask(df: pd.DataFrame):
    # 후보 레벨 칼럼들(있으면 사용, 없으면 전체 True)
    level_cols = [c for c in df.columns if "level" in c.lower()]
    if not level_cols: 
        return np.ones(len(df), dtype=bool)
    # 동일 레벨 판단: 간단히 "이벤트 그룹 단위에서 레벨 값이 비어있지 않은 행"으로 제한
    # (진짜 strict 동일 레벨 비교가 필요하면 특정 칼럼명을 알려줘)
    m = ~df[level_cols].isna().all(axis=1)
    return m.values

def substr_any(s: str, keys):
    s = (s or "").lower()
    return any(k in s for k in keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="labeled signals csv (from label_events_side.py)")
    ap.add_argument("--out", required=True, help="output filtered csv path")
    ap.add_argument("--dist-max", type=float, default=0.00025, help="ratio scale (e.g., 0.00025 = 0.025%)")
    ap.add_argument("--include-substr", default="support", help="comma separated substrings to require in event")
    ap.add_argument("--require-same-level", action="store_true", help="filter rows with non-empty level columns only")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    if "event" not in df.columns:
        raise SystemExit("input must have 'event' column")
    if "side" not in df.columns:
        # 라벨링 미적용 파일도 허용(간단 추론)
        df["side"] = df["event"].astype(str).str.lower().map(lambda e: "support" if "support" in e or "sup" in e else ("resistance" if "resist" in e or "breakout" in e else "neutral"))

    # ts 정규화
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    # support 라인만
    sub = df[df["side"] == "support"].copy()

    # 이벤트명 부분문자열 필터(기본: 'support' 포함)
    keys = [k.strip().lower() for k in (args.include_substr or "").split(",") if k.strip()]
    if keys:
        sub = sub[sub["event"].astype(str).str.lower().map(lambda s: substr_any(s, keys))]

    # distance 필터(있으면 적용, 없으면 스킵 경고)
    if "distance_pct" in sub.columns:
        mask, scale = auto_scale_and_mask(sub["distance_pct"], args.dist_max)
        sub = sub[mask].copy()
        print(f"[FILTER] distance_pct filter ({scale}) thr={args.dist_max}: -> {len(sub)} rows")
    else:
        print("[FILTER] distance_pct column missing; skip dist filter")

    # 동일 레벨 강제(레벨 칼럼 있는 경우)
    if args.require_same_level:
        m = same_level_mask(sub)
        before = len(sub); sub = sub[m].copy()
        print(f"[FILTER] require_same_level: {before} -> {len(sub)}")

    sub.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OUT] saved -> {args.out} (rows={len(sub)})")

if __name__ == "__main__":
    main()