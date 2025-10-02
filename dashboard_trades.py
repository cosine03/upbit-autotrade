# dashboard_trades.py
# Streamlit dashboard for visualizing source usage (WS/CSV/REST) with colors

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(ROOT, "logs", "paper")
OPEN_CSV = os.path.join(LOGS_DIR, "trades_open.csv")
CLOSE_CSV = os.path.join(LOGS_DIR, "trades_closed.csv")

# Color map for sources
SRC_COLOR = {
    "ws":   "#12b886",  # teal
    "csv":  "#4dabf7",  # blue
    "rest": "#ff922b",  # orange
    "":     "#ced4da",  # empty
    None:   "#ced4da",
}

def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df

def _parse_dt_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            pass
    return df

def load_data():
    df_open  = _read_csv_safe(OPEN_CSV)
    df_close = _read_csv_safe(CLOSE_CSV)

    # expected columns:
    # trades_open:  opened_at,symbol,event,side,level,expire_at,fee,entry_src?
    # trades_closed: opened_at,symbol,event,side,level,closed_at,entry_price,exit_price,pnl,reason,fee,exit_src?
    # parse times
    for c in ("opened_at","expire_at"):
        df_open = _parse_dt_col(df_open, c)
    for c in ("opened_at","closed_at"):
        df_close = _parse_dt_col(df_close, c)

    # normalize src columns
    if "entry_src" not in df_open.columns:
        # older logs may not have this column; fill empty
        df_open["entry_src"] = ""

    if "exit_src" not in df_close.columns:
        df_close["exit_src"] = ""

    # basic sorts (latest first)
    if "opened_at" in df_open.columns:
        df_open = df_open.sort_values("opened_at", ascending=False)
    if "closed_at" in df_close.columns:
        df_close = df_close.sort_values("closed_at", ascending=False)

    return df_open, df_close

def bar_by_source(df: pd.DataFrame, src_col: str, title: str):
    s = df[src_col].fillna("").replace("", "n/a")
    counts = s.value_counts().rename_axis("source").reset_index(name="count")
    if counts.empty:
        st.info("No data to plot.")
        return
    fig = px.bar(
        counts,
        x="source", y="count",
        title=title,
        text="count",
    )
    # apply colors if keys known
    colors = [SRC_COLOR.get(k if k != "n/a" else "", "#adb5bd") for k in counts["source"]]
    fig.update_traces(marker_color=colors, textposition="outside")
    fig.update_layout(yaxis_title="count", xaxis_title="source", uniformtext_minsize=10, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)

def plotly_table_with_colors(df: pd.DataFrame, src_col: str, title: str, cols: list[str]):
    if df.empty:
        st.info("No rows.")
        return

    # slice
    show = df[cols].copy()

    # build per-cell colors (only for src_col)
    fill_colors = []
    for _, row in show.iterrows():
        row_colors = []
        for c in cols:
            if c == src_col:
                row_colors.append(SRC_COLOR.get(row[c], "#e9ecef"))
            else:
                row_colors.append("#ffffff")
        fill_colors.append(row_colors)

    # transpose colors for go.Table (expects column-wise)
    fill_colors_cols = list(map(list, zip(*fill_colors)))

    # Convert datetimes to strings for pretty display
    def fmt(v):
        if isinstance(v, pd.Timestamp):
            if pd.isna(v):
                return ""
            return v.tz_convert("UTC").isoformat()
        return "" if pd.isna(v) else v

    values = [show[c].map(fmt).to_list() for c in cols]

    header_colors = ["#f1f3f5"] * len(cols)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=cols,
                    fill_color=header_colors,
                    align="left",
                    font=dict(color="#212529", size=12),
                ),
                cells=dict(
                    values=values,
                    align="left",
                    fill_color=fill_colors_cols,
                    font=dict(color="#212529", size=12),
                    height=26,
                ),
            )
        ]
    )
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=36, b=0))
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="PaperTrader – Source View", layout="wide")
    st.title("PaperTrader – Source usage (WS / CSV / REST)")

    df_open, df_close = load_data()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Open rows", len(df_open))
    with c2:
        st.metric("Closed rows", len(df_close))
    with c3:
        ws_opens = int((df_open["entry_src"] == "ws").sum()) if "entry_src" in df_open else 0
        st.metric("WS entries", ws_opens)
    with c4:
        ws_exits = int((df_close["exit_src"] == "ws").sum()) if "exit_src" in df_close else 0
        st.metric("WS exits", ws_exits)

    st.subheader("Source distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        bar_by_source(df_open,  "entry_src", "ENTRY source distribution")
    with col_b:
        bar_by_source(df_close, "exit_src",  "EXIT source distribution")

    st.subheader("Recent OPENs (colored by entry_src)")
    open_cols = []
    for c in ("opened_at","symbol","event","side","level","expire_at","fee","entry_src"):
        if c in df_open.columns:
            open_cols.append(c)
    plotly_table_with_colors(df_open.head(200), "entry_src", "trades_open (latest 200)", open_cols)

    st.subheader("Recent CLOSED (colored by exit_src)")
    close_cols = []
    for c in ("closed_at","symbol","pnl","entry_price","exit_price","reason","fee","exit_src","opened_at","event","side","level"):
        if c in df_close.columns:
            close_cols.append(c)
    plotly_table_with_colors(df_close.head(200), "exit_src", "trades_closed (latest 200)", close_cols)

    st.caption("Color legend – ws: teal, csv: blue, rest: orange, blank: gray.")

if __name__ == "__main__":
    main()