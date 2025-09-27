@'
import pandas as pd, os, glob

sig = pd.read_csv(r".\logs\signals_binance_enriched_breakout_resist.csv")
sig["ts"] = pd.to_datetime(sig["ts"], utc=True, errors="coerce")

print("== signals window ==")
print("min:", sig["ts"].min(), "max:", sig["ts"].max())
print("symbols:", sorted(sig["symbol"].unique().tolist()))

have = {os.path.basename(p).split("-")[0]: p for p in glob.glob(r".\data\ohlcv_binance\*-15m.csv")}

rows = []
for sym, g in sig.groupby("symbol"):
    p = have.get(sym)
    if not p:
        rows.append((sym, "MISSING_OHLCV", None, None, g["ts"].min(), g["ts"].max(), 0))
        continue
    o = pd.read_csv(p)
    o["ts"] = pd.to_datetime(o["ts"], utc=True, errors="coerce")
    max_o = o["ts"].max()
    min_o = o["ts"].min()
    ok2h = int((g["ts"] <= (max_o - pd.Timedelta(hours=2))).sum())
    rows.append((sym, "OK", min_o, max_o, g["ts"].min(), g["ts"].max(), ok2h))

out = pd.DataFrame(rows, columns=[
    "symbol","status","ohlcv_min","ohlcv_max","sig_min","sig_max","signals_with_2h_future"
])
print("\n== coverage by symbol ==")
print(out.to_string(index=False))
'@ | python -