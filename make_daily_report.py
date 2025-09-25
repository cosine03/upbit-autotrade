# -*- coding: utf-8 -*-
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in1", required=True, help="summary csv for breakout-only")
    ap.add_argument("--in2", required=True, help="summary csv for boxin-linebreak subset")
    ap.add_argument("--out", required=True, help="output xlsx")
    ap.add_argument("--tag", default="", help="date tag to annotate")
    args = ap.parse_args()

    b = pd.read_csv(args.in1)
    s = pd.read_csv(args.in2)

    keep = ["event","expiry_h","trades","win_rate","avg_net","median_net","total_net"]
    b = b[keep].copy(); s = s[keep].copy()
    b["strategy"] = "breakout_only"; s["strategy"] = "boxin_linebreak"
    both = pd.concat([b,s], ignore_index=True)

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        b.to_excel(writer, sheet_name="breakout_only", index=False)
        s.to_excel(writer, sheet_name="boxin_linebreak", index=False)
        both.to_excel(writer, sheet_name="combined", index=False)

    # pivot & charts
    pivot_avg = both.pivot_table(index=["expiry_h"], columns="strategy", values="avg_net", aggfunc="mean").reset_index()
    pivot_tot = both.pivot_table(index=["expiry_h"], columns="strategy", values="total_net", aggfunc="sum").reset_index()
    pivot_wr  = both.pivot_table(index=["expiry_h"], columns="strategy", values="win_rate", aggfunc="mean").reset_index()

    base = os.path.splitext(args.out)[0]
    charts = [
        (pivot_avg, f"AVG_NET by expiry ({args.tag})", base+"_avg_net.png"),
        (pivot_tot, f"TOTAL_NET by expiry ({args.tag})", base+"_total_net.png"),
        (pivot_wr,  f"WIN_RATE by expiry ({args.tag})", base+"_win_rate.png")
    ]

    for dfp, title, path in charts:
        plt.figure(figsize=(8,5))
        for col in dfp.columns[1:]:
            plt.bar([f"{x} {col}" for x in dfp["expiry_h"]], dfp[col])
        plt.title(title); plt.xlabel("expiry_h x strategy"); plt.ylabel(title.split()[0].lower())
        plt.tight_layout(); plt.savefig(path); plt.close()

    wb = load_workbook(args.out)
    ws = wb.create_sheet("charts")
    ws["A1"] = f"Tag: {args.tag}"
    row = 3
    for _,_,p in charts:
        if os.path.exists(p):
            img = XLImage(p); img.anchor = f"A{row}"
            ws.add_image(img); row += 25
    wb.save(args.out)
    print("[REPORT] saved ->", args.out)

if __name__ == "__main__":
    main()