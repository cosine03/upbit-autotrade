# normalize_ohlcv.py
import os, glob
import pandas as pd

SRC = r".\logs\ohlcv\compat"
DST = r".\logs\ohlcv\compat_clean"
TF  = "15m"

os.makedirs(DST, exist_ok=True)

# 컬럼 후보 맵 (파일마다 이름이 다를 수 있으니 넓게 커버)
COLMAPS = [
    {"ts":"ts","open":"open","high":"high","low":"low","close":"close","volume":"volume"},
    {"timestamp":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume"},
    {"time":"ts","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"},
]

def normalize_one(path):
    name = os.path.basename(path)  # ex) KRW-ETH-15m.csv
    sym  = name.replace(f"-{TF}.csv","").replace(f"_{TF}.csv","")
    df = pd.read_csv(path)

    # 컬럼명 매핑
    cols = {c.lower():c for c in df.columns}
    used = None
    for cmap in COLMAPS:
        if all(k in cols for k in cmap.keys()):
            used = { cmap[k]: cols[k] for k in cmap.keys() }
            break
    if used is None:
        raise RuntimeError(f"Unrecognized columns: {df.columns.tolist()} @ {path}")

    df = df.rename(columns=used)[["ts","open","high","low","close"] + (["volume"] if "volume" in used else [])]

    # ts 파싱 → UTC → 15분 스냅 → 문자열 "+00:00" 포맷
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.assign(ts=ts).dropna(subset=["ts"])
    # 15분 경계로 바닥(floor); 필요하면 round('15T')로 교체 가능
    df["ts"] = df["ts"].dt.floor("15T")
    # 중복 ts 제거(최근값 우선)
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    # 문자열 포맷: 2025-09-19 04:15:00+00:00
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+00)(00)$", r"\1:\2", regex=True)
    # 혹시 밀리초열 있으면 제거
    for c in ("Open","High","Low","Close","open","high","low","close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = os.path.join(DST, f"{sym}_{TF}.csv")
    df.to_csv(out, index=False)
    return out, len(df)

def main():
    files = sorted(glob.glob(os.path.join(SRC, f"*-{TF}.csv")))
    ok = 0
    for f in files:
        try:
            out, n = normalize_one(f)
            print(f"[NORM] {os.path.basename(f)} -> {os.path.basename(out)} (rows={n})")
            ok += 1
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
    print(f"[DONE] normalized {ok}/{len(files)} files")

if __name__ == "__main__":
    main()