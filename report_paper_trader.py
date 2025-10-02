# report_paper_trader.py
# -*- coding: utf-8 -*-
import argparse, csv, os, sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

ROOT = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(ROOT, "logs", "paper")
CLOSED = os.path.join(LOGS_DIR, "trades_closed.csv")
REJECTS = os.path.join(LOGS_DIR, "rejects.csv")

def parse_dt(s: str) -> Optional[datetime]:
    try:
        # supports: 2025-10-02T06:55:54.418252+00:00
        return datetime.fromisoformat(s)
    except Exception:
        return None

def to_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def read_csv(path: str) -> List[List[str]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))

def f6(x):
    try: return f"{float(x):.6f}"
    except: return ""

def load_closed(since: datetime, until: datetime) -> List[Dict]:
    rows = read_csv(CLOSED)
    out = []
    if not rows: return out
    header = rows[0]
    idx = {k:i for i,k in enumerate(header)}
    required = ["opened_at","symbol","event","side","level",
                "closed_at","entry_price","exit_price","pnl","reason","fee"]
    if not all(k in idx for k in required):
        return out
    # optional columns
    src_entry_idx = idx.get("entry_src")
    src_exit_idx  = idx.get("exit_src")

    for r in rows[1:]:
        try:
            closed_at = parse_dt(r[idx["closed_at"]])
            if not closed_at: continue
            if not (to_utc(since) <= closed_at <= to_utc(until)): continue
            d = {
                "opened_at": r[idx["opened_at"]],
                "symbol":    r[idx["symbol"]],
                "event":     r[idx["event"]],
                "side":      r[idx["side"]],
                "level":     r[idx["level"]],
                "closed_at": r[idx["closed_at"]],
                "entry_price": r[idx["entry_price"]] or "",
                "exit_price":  r[idx["exit_price"]] or "",
                "pnl":         r[idx["pnl"]] or "",
                "reason":      r[idx["reason"]],
                "fee":         r[idx["fee"]] or "",
                "entry_src":   (r[src_entry_idx] if src_entry_idx is not None and src_entry_idx < len(r) else ""),
                "exit_src":    (r[src_exit_idx]  if src_exit_idx  is not None and src_exit_idx  < len(r) else ""),
            }
            out.append(d)
        except Exception:
            continue
    return out

def load_rejects(since: datetime, until: datetime) -> List[Dict]:
    rows = read_csv(REJECTS)
    out = []
    if not rows: return out
    header = rows[0]
    idx = {k:i for i,k in enumerate(header)}
    req = ["ts","reason"]
    if not all(k in idx for k in req):
        return out
    for r in rows[1:]:
        try:
            ts = parse_dt(r[idx["ts"]])
            if not ts: continue
            if not (to_utc(since) <= ts <= to_utc(until)): continue
            out.append({
                "ts": r[idx["ts"]],
                "reason": r[idx["reason"]],
                "symbol": r[idx.get("symbol", -1)] if "symbol" in idx else "",
            })
        except Exception:
            continue
    return out

def summarize(closed: List[Dict], rejects: List[Dict]) -> str:
    n = len(closed)
    wins = sum(1 for d in closed if d.get("pnl") not in ("","None") and float(d["pnl"]) > 0)
    losses = sum(1 for d in closed if d.get("pnl") not in ("","None") and float(d["pnl"]) <= 0)
    win_rate = (wins / n * 100.0) if n else 0.0

    net_sum = sum(float(d["pnl"]) for d in closed if d.get("pnl") not in ("","None"))
    avg_pnl = (net_sum / n) if n else 0.0

    by_src_entry: Dict[str,int] = {}
    by_src_exit:  Dict[str,int] = {}
    for d in closed:
        se = (d.get("entry_src") or "na").lower()
        sx = (d.get("exit_src")  or "na").lower()
        by_src_entry[se] = by_src_entry.get(se,0)+1
        by_src_exit[sx]  = by_src_exit.get(sx,0)+1

    # top symbols
    by_sym: Dict[str, Dict[str,float]] = {}
    for d in closed:
        s = d["symbol"]
        by_sym.setdefault(s, {"n":0,"pnl":0.0})
        by_sym[s]["n"] += 1
        if d.get("pnl") not in ("","None"):
            by_sym[s]["pnl"] += float(d["pnl"])
    top_syms = sorted(by_sym.items(), key=lambda kv: (kv[1]["pnl"], kv[1]["n"]), reverse=True)[:5]

    # reject reasons
    rej_count: Dict[str,int] = {}
    for r in rejects:
        rej_count[r["reason"]] = rej_count.get(r["reason"],0)+1
    top_rej = sorted(rej_count.items(), key=lambda kv: kv[1], reverse=True)[:5]

    lines = []
    lines.append("# Paper Trader Daily Report")
    lines.append("")
    lines.append(f"- Closed positions: **{n}** | Wins: **{wins}** Losses: **{losses}** | Win rate: **{win_rate:.1f}%**")
    lines.append(f"- Net PnL (sum): **{net_sum:+.4f}** | Avg PnL: **{avg_pnl:+.4f}**")
    lines.append("")
    lines.append("## Entry source breakdown")
    for k,v in sorted(by_src_entry.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Exit source breakdown")
    for k,v in sorted(by_src_exit.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Top symbols (by net PnL)")
    if top_syms:
        for s,agg in top_syms:
            lines.append(f"- {s}: n={agg['n']}, pnl={agg['pnl']:+.4f}")
    else:
        lines.append("- (no data)")
    lines.append("")
    lines.append("## Top reject reasons")
    if top_rej:
        for r,cnt in top_rej:
            lines.append(f"- {r}: {cnt}")
    else:
        lines.append("- (no data)")
    lines.append("")
    lines.append("— Generated automatically.")
    return "\n".join(lines)

def write_closed_csv(outdir: str, closed: List[Dict], tag: Optional[str]):
    path = os.path.join(outdir, f"report_closed_{(tag or 'last24h')}.csv")
    fields = ["opened_at","symbol","event","side","level","closed_at",
              "entry_price","exit_price","pnl","reason","fee","entry_src","exit_src"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in closed:
            w.writerow({k: d.get(k,"") for k in fields})
    return path

def main():
    ap = argparse.ArgumentParser(description="Paper Trader Daily Report")
    ap.add_argument("--since", help="ISO8601 (UTC). If omitted, uses --range")
    ap.add_argument("--until", help="ISO8601 (UTC). Default: now")
    ap.add_argument("--range", default="24h", help="e.g. 24h, 12h; used if --since not provided")
    ap.add_argument("--out", default=LOGS_DIR, help="output directory")
    ap.add_argument("--tag", default="last24h", help="AM/PM or custom tag for filenames")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    if args.since:
        since = to_utc(parse_dt(args.since))
    else:
        # parse range like "24h"
        hours = int(args.range.lower().replace("h",""))
        since = now - timedelta(hours=hours)
    until = to_utc(parse_dt(args.until)) if args.until else now

    closed = load_closed(since, until)
    rejects = load_rejects(since, until)
    md = summarize(closed, rejects)

    os.makedirs(args.out, exist_ok=True)
    md_path = os.path.join(args.out, f"report_{args.tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    csv_path = write_closed_csv(args.out, closed, args.tag)
    print(f"[OK] Report saved: {md_path}\n[OK] Closed CSV: {csv_path}\nClosed={len(closed)}, Rejects={len(rejects)}")

if __name__ == "__main__":
    main()