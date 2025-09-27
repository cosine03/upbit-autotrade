import pandas as pd, sys

src = r".\logs\signals_binance_as_upbit_enriched.csv"
dst = r".\logs\signals_binance_as_upbit_enriched_breakout.csv"

df = pd.read_csv(src)
df = df[df["event"].isin(["box_breakout", "line_breakout"])]

df.to_csv(dst, index=False)
print("saved breakout-only ->", dst, "rows=", len(df))