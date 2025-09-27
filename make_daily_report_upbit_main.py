# make_daily_report_upbit_main.py
# Usage: python make_daily_report_upbit_main.py --root . --tag AM
import argparse, os, glob, json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

def latest_daily_dir(root, tag):
    # logs/daily/YYYY-MM-DD_TAG 중 가장 최신을 선택
    pat = os.path.join(root, "logs", "daily", f"20??-??-??_{tag}")
    dirs = sorted(glob.glob(pat))
    if not dirs:
        raise SystemExit(f"[ERR] daily dir not found for tag={tag} in {pat}")
    return dirs[-1]

def safe_read_csv(path):
    if not os.path.exists(path): return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def plot_and_save(df, x, y, title, outpng):
    if df is None or df.empty: return
    plt.figure()
    df.plot(x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="AM")  # AM or PM
    args = ap.parse_args()

    outdir = latest_daily_dir(args.root, args.tag)
    print("[REPORT] target daily dir:", outdir)

    # 1) 핵심 파일 로드
    path_summary_main = os.path.join(outdir, "bt_breakout_only", "bt_tv_events_stats_summary.csv")
    path_trades_main  = os.path.join(outdir, "bt_breakout_only", "bt_tv_events_trades.csv")
    path_signals_main = os.path.join(outdir, "signals_breakout_only.csv")
    path_params       = os.path.join(outdir, "dynamic_params.json")

    df_sum = safe_read_csv(path_summary_main)
    df_trd = safe_read_csv(path_trades_main)
    df_sig = safe_read_csv(path_signals_main)
    params = {}
    if os.path.exists(path_params):
        with open(path_params, "r", encoding="utf-8") as f:
            params = json.load(f)

    # 2) 요약 KPI
    kpis = {}
    if df_sum is not None and not df_sum.empty:
        # 메인 전략은 event ∈ {box_breakout,line_breakout} 집계로 가정
        sub = df_sum[df_sum["event"].isin(["box_breakout","line_breakout"])].copy()
        if not sub.empty:
            sub = sub.groupby("expiry_h", as_index=False).agg(
                trades=("trades","sum"),
                win_rate=("win_rate","mean"),
                avg_net=("avg_net","mean"),
                median_net=("median_net","mean"),
                total_net=("total_net","sum"),
            ).sort_values("expiry_h")
            kpis["summary_by_expiry"] = sub
            print("\n[SUMMARY by expiry]")
            print(sub.to_string(index=False))
    if df_sig is not None and not df_sig.empty:
        kpis["signals_count"] = len(df_sig)

    # 3) 차트 저장
    if "summary_by_expiry" in kpis:
        sub = kpis["summary_by_expiry"]
        plot_and_save(sub, "expiry_h", "win_rate",      "Win Rate by Expiry (Main: Upbit/Upbit)", os.path.join(outdir, "daily_report_win_rate.png"))
        plot_and_save(sub, "expiry_h", "avg_net",       "Avg Net by Expiry (Main: Upbit/Upbit)", os.path.join(outdir, "daily_report_avg_net.png"))
        plot_and_save(sub, "expiry_h", "total_net",     "Total Net by Expiry (Main: Upbit/Upbit)", os.path.join(outdir, "daily_report_total_net.png"))
    # 트레이드 equity 곡선(있으면)
    if df_trd is not None and not df_trd.empty:
        # 간단 equity
        df_trd = df_trd.sort_values(["expiry_h"]).copy()
        for eh, g in df_trd.groupby("expiry_h"):
            g = g.copy()
            if "net" in g.columns:
                g["equity"] = g["net"].cumsum()
                plt.figure()
                g["equity"].reset_index(drop=True).plot()
                plt.title(f"Equity Curve (expiry={eh}h)")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"daily_report_equity_main_{eh}.png"))
                plt.close()

    # 4) 엑셀 저장
    xlsx = os.path.join(outdir, "daily_report_upbit_main.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as wr:
        if df_sum is not None: df_sum.to_excel(wr, "stats_summary_raw", index=False)
        if df_trd is not None: df_trd.to_excel(wr, "trades", index=False)
        if df_sig is not None: df_sig.to_excel(wr, "signals", index=False)
        if "summary_by_expiry" in kpis: kpis["summary_by_expiry"].to_excel(wr, "summary_by_expiry", index=False)
        pd.DataFrame([{
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "signals_count": kpis.get("signals_count", 0),
            "dist_max_ratio": params.get("dist_max",{}).get("value_ratio"),
            "dist_max_percent": params.get("dist_max",{}).get("value_percent"),
        }]).to_excel(wr, "meta", index=False)
    print("[REPORT] saved ->", xlsx)

if __name__ == "__main__":
    main()