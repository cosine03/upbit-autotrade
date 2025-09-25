# -*- coding: utf-8 -*-
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse, os as _os
import pandas as pd
import numpy as np

def compute_equity_and_mdd(trades: pd.DataFrame, group_cols=("expiry_h",)):
    out = {}
    if "net" not in trades.columns:
        return out
    if "ts_entry" in trades.columns:
        trades["ts_entry"] = pd.to_datetime(trades["ts_entry"], errors="coerce", utc=True)
        trades = trades.sort_values(["expiry_h","ts_entry"])
    for keys, g in trades.groupby(list(group_cols)):
        eq = g["net"].cumsum().reset_index(drop=True)
        peak = eq.cummax()
        dd = eq - peak
        mdd = dd.min() if len(dd) else 0.0
        equity = pd.DataFrame({"step": np.arange(len(eq)), "equity": eq.values})
        out[keys if isinstance(keys, tuple) else (keys,)] = (equity, float(mdd))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="bt_tv_events_stats_summary.csv (single strategy)")
    ap.add_argument("--trades",  required=True, help="bt_tv_events_trades.csv")
    ap.add_argument("--out",     required=True, help="output xlsx")
    ap.add_argument("--tag",     default="")
    ap.add_argument("--strategy", default="boxin_linebreak")
    args = ap.parse_args()

    s = pd.read_csv(args.summary)
    t = pd.read_csv(args.trades)

    # keep core cols
    keep = ["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"]
    s = s[keep].copy()

    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        s.to_excel(w, sheet_name="summary", index=False)
        t.to_excel(w, sheet_name="trades",  index=False)

        # pivots by expiry
        pv = s.pivot_table(index="expiry_h", values=["trades","win_rate","avg_net","total_net"], aggfunc={"trades":"sum","win_rate":"mean","avg_net":"mean","total_net":"sum"}).reset_index()
        pv.to_excel(w, sheet_name="pivot_by_expiry", index=False)

    # charts
    import matplotlib.pyplot as plt
    from openpyxl import load_workbook
    from openpyxl.drawing.image import Image as XLImage

    base = _os.path.splitext(args.out)[0]
    p1, p2, p3 = base+"_avg_net.png", base+"_total_net.png", base+"_win_rate.png"

    # simple bars
    fig, ax = plt.subplots(figsize=(7,4))
    plt.bar(pv["expiry_h"].astype(str), pv["avg_net"]); ax.set_title(f"AVG_NET by expiry ({args.tag})"); ax.set_xlabel("expiry_h"); ax.set_ylabel("avg_net"); plt.tight_layout(); plt.savefig(p1); plt.close()

    fig, ax = plt.subplots(figsize=(7,4))
    plt.bar(pv["expiry_h"].astype(str), pv["total_net"]); ax.set_title(f"TOTAL_NET by expiry ({args.tag})"); ax.set_xlabel("expiry_h"); ax.set_ylabel("total_net"); plt.tight_layout(); plt.savefig(p2); plt.close()

    fig, ax = plt.subplots(figsize=(7,4))
    plt.bar(pv["expiry_h"].astype(str), pv["win_rate"]); ax.set_title(f"WIN_RATE by expiry ({args.tag})"); ax.set_xlabel("expiry_h"); ax.set_ylabel("win_rate"); plt.tight_layout(); plt.savefig(p3); plt.close()

    # equity & MDD
    emap = compute_equity_and_mdd(t)
    eq_imgs = []
    for (expiry,), (eqdf, mdd) in emap.items():
        fn = f"{base}_equity_{expiry}h.png"
        plt.figure(figsize=(7,3.5))
        plt.plot(eqdf["step"], eqdf["equity"])
        plt.title(f"Equity Curve - {args.strategy} - {expiry}h | MDD={mdd:.4f}")
        plt.xlabel("trade idx"); plt.ylabel("cum net")
        plt.tight_layout(); plt.savefig(fn); plt.close()
        eq_imgs.append((fn, expiry, mdd))

    # insert images
    wb = load_workbook(args.out)
    ws = wb.create_sheet("charts")
    ws["A1"] = f"Strategy: {args.strategy} | Tag: {args.tag}"
    row = 3
    for p in (p1, p2, p3):
        if _os.path.exists(p):
            img = XLImage(p); img.anchor = f"A{row}"; ws.add_image(img); row += 22

    # equity/mdd sheet
    ws2 = wb.create_sheet("equity_mdd")
    ws2["A1"] = "expiry_h"; ws2["B1"] = "MDD"
    r = 2
    for fn, expiry, mdd in eq_imgs:
        ws2[f"A{r}"] = float(expiry)
        ws2[f"B{r}"] = float(mdd)
        if _os.path.exists(fn):
            img = XLImage(fn); img.anchor = f"D{r}"; ws2.add_image(img)
        r += 18

    wb.save(args.out)
    print("[REPORT] saved ->", args.out)

if __name__ == "__main__":
    main()