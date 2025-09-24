# utils/agg.py
import pandas as pd

def summarize_trades(df: pd.DataFrame, by=("event", "expiry_h")) -> pd.DataFrame:
    """
    pandas 버전 차이에 안전한 집계 루틴.
    - trades: 건수 (NaN 포함 안전하게 size())
    - avg_net, median_net, total_net: 표준 통계
    - win_rate: (net > 0) 비율
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[*by, "trades", "avg_net", "median_net", "total_net", "win_rate"])

    # 그룹
    g = df.groupby(list(by), dropna=False)

    # 건수
    trades_ct = g.size().reset_index(name="trades")

    # 수익 통계
    stats = (
        g["net"]
        .agg(["mean", "median", "sum"])
        .reset_index()
        .rename(columns={"mean": "avg_net", "median": "median_net", "sum": "total_net"})
    )

    # 승률
    win_rate = g["net"].apply(lambda s: (s > 0).mean()).reset_index(name="win_rate")

    # 합치기
    out = trades_ct.merge(stats, on=list(by), how="left").merge(win_rate, on=list(by), how="left")

    # 결측 방지
    return out.fillna(
        {"trades": 0, "avg_net": 0.0, "median_net": 0.0, "total_net": 0.0, "win_rate": 0.0}
    )
