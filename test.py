import pandas as pd

df = pd.read_csv(r".\logs\signals_tv_enriched.csv")

if "distance_pct" in df.columns:
    print(df["distance_pct"].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]))
    print("\n상위 20개 값:")
    print(df["distance_pct"].nlargest(20).to_list())