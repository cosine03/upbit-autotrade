#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, subprocess, sys, os, csv
from pathlib import Path

def file_has_column(csv_path: Path, colname: str) -> bool:
    try:
        with csv.open(csv_path, "r", newline="") as f:  # type: ignore[attr-defined]
            pass
    except Exception:
        # fallback without csv.open (py3.11+ only). Use classic open.
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            return colname in header
    return False  # shouldn’t get here

def csv_has_col(csv_path: Path, colname: str) -> bool:
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            return colname in header
    except Exception:
        return False

def run(cmd: list[str], title: str):
    print(f"[PIPE] {title}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print(f"[PIPE][ERROR] {title} failed with code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)

def main():
    p = argparse.ArgumentParser(
        description="TV signals 파이프라인 (enrich → backtest) 원클릭 실행기"
    )
    p.add_argument("signals", help="signals_tv.csv 또는 signals_tv_enriched.csv 경로")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--expiries", default="0.5h,1h,2h")
    p.add_argument("--tp", type=float, default=1.5)
    p.add_argument("--sl", type=float, default=0.8)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--dist-max", type=float, default=0.02)
    p.add_argument("--procs", type=int, default=24)
    p.add_argument("--outdir", default="./logs/bt_tv_pipeline_run")

    # 엔리치 스크립트/옵션(프로젝트마다 이름 다를 수 있어 유연하게)
    p.add_argument("--enrich-script", default="tv_level_estimator.py",
                   help="enriched CSV 생성 스크립트 파일명 (예: tv_level_estimator.py)")
    p.add_argument("--enrich-out", default="./logs/signals_tv_enriched.csv")
    p.add_argument("--skip-enrich", action="store_true",
                   help="입력 CSV가 이미 enriched라면 강제 스킵")

    # 백테스트 스크립트 파일명(이미 쓰고 있는 것)
    p.add_argument("--bt-script", default="backtest_tv_events_mp.py")

    args = p.parse_args()

    sig_path = Path(args.signals).resolve()
    if not sig_path.exists():
        print(f"[PIPE][ERROR] signals 파일이 없음: {sig_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Enrich 필요 여부 판단
    enriched_path = Path(args.enrich_out).resolve()
    need_enrich = True
    if args.skip_enrich:
        need_enrich = False
    else:
        # 이미 enriched 파일을 입력으로 준 경우거나 distance_pct가 있으면 생략
        if "enriched" in sig_path.name.lower() or csv_has_col(sig_path, "distance_pct"):
            enriched_path = sig_path
            need_enrich = False

    # 2) Enrich 실행
    if need_enrich:
        # 추정: enrich 스크립트가 다음 인자를 받는 형태
        #   python tv_level_estimator.py <signals_csv> --timeframe 15m --out <enriched_csv>
        # 프로젝트 환경에 따라 이름/인자 다르면 --enrich-script / --enrich-out로 맞추면 됨
        run([
            sys.executable, args.enrich_script,
            str(sig_path),
            "--timeframe", args.timeframe,
            "--out", str(enriched_path)
        ], "enrich")

        # 기본 검증: distance_pct가 생겼는지
        if not csv_has_col(enriched_path, "distance_pct"):
            print("[PIPE][WARN] enriched 파일에 distance_pct 컬럼이 보이지 않습니다. "
                  "enrich 스크립트 인자를 확인하세요.")

    # 3) Backtest 실행
    run([
        sys.executable, args.bt_script,
        str(enriched_path),
        "--timeframe", args.timeframe,
        "--expiries", args.expiries,
        "--tp", str(args.tp),
        "--sl", str(args.sl),
        "--fee", str(args.fee),
        "--dist-max", str(args.dist_max),
        "--procs", str(args.procs),
        "--outdir", str(outdir)
    ], "backtest")

    print("\n[PIPE] 완료! 결과 폴더:", outdir)

if __name__ == "__main__":
    main()