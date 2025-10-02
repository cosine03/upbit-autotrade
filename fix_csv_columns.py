# fix_csv_columns.py
import csv, shutil, sys, os

def fix_trades_closed(path: str):
    tmp = path + ".tmp"
    with open(path, newline="", encoding="utf-8") as f, open(tmp, "w", newline="", encoding="utf-8") as g:
        rdr = csv.reader(f)
        w = csv.writer(g)
        header = next(rdr)
        # 최신 헤더 기준 (12칸): [..., 'pnl','reason','fee','exit_src']
        # 실제 파일 헤더에 맞춰 그대로 유지
        w.writerow(header)
        H = len(header)
        for r in rdr:
            if len(r) == H:
                w.writerow(r)
                continue
            # 아주 예전 8칸 형식: [... closed_at, reason, fee]
            if len(r) == 8:
                # r = [opened_at, symbol, event, side, level, closed_at, reason, fee]
                fixed = r[:6] + ["", "", ""] + r[6:] + [""]  # entry,exit,pnl 추가 + exit_src 추가
                # fixed 길이 보정
                if len(fixed) < H: fixed += [""] * (H - len(fixed))
                elif len(fixed) > H: fixed = fixed[:H]
                w.writerow(fixed)
                continue
            # 11칸(= exit_src 한 칸 부족)
            if len(r) == 11:
                w.writerow(r + [""])
                continue
            # 그 외: 헤더 길이에 맞춰 패딩/자르기
            if len(r) < H:
                r = r + [""] * (H - len(r))
            elif len(r) > H:
                r = r[:H]
            w.writerow(r)
    shutil.move(path, path + ".bak")
    shutil.move(tmp, path)
    print(f"fixed {path} (backup: {path}.bak)")

def fix_generic_trim_or_pad(path: str):
    tmp = path + ".tmp"
    with open(path, newline="", encoding="utf-8") as f, open(tmp, "w", newline="", encoding="utf-8") as g:
        rdr = csv.reader(f)
        w = csv.writer(g)
        header = next(rdr)
        H = len(header)
        w.writerow(header)
        for r in rdr:
            if len(r) < H:
                r = r + [""] * (H - len(r))
            elif len(r) > H:
                # 초과 칸은 버리고 헤더 길이에 맞춤 (필요하면 마지막 칸에 합치는 로직으로 바꿀 수 있음)
                r = r[:H]
            w.writerow(r)
    shutil.move(path, path + ".bak")
    shutil.move(tmp, path)
    print(f"fixed {path} (backup: {path}.bak)")

if __name__ == "__main__":
    # 원하는 파일만 넣어도 됩니다.
    paths = [
        "logs/paper/trades_closed.csv",
        "logs/paper/trades.csv",
    ]
    for p in paths:
        if not os.path.exists(p):
            print(f"skip (not found): {p}")
            continue
        if p.endswith("trades_closed.csv"):
            fix_trades_closed(p)
        else:
            fix_generic_trim_or_pad(p)
    print("done.")