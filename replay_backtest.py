# replay_backtest.py
# -*- coding: utf-8 -*-
import csv, os, sys, argparse
from datetime import datetime, timezone, timedelta
from bisect import bisect_left
from collections import defaultdict

def parse_dt(s):
    # signals/prices 둘 다 UTC ISO 가정
    s = s.strip().replace("Z","+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def load_prices(prices_csv, universe=None):
    # CSV 헤더: ts,symbol,price,src  (price_feeder_upbit.py 출력)
    by_sym = defaultdict(list)  # sym -> [(ts,price)]
    with open(prices_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            sym = r.get("symbol","").strip().upper()
            if not sym: continue
            if universe and sym not in universe: continue
            try:
                ts = parse_dt(r["ts"])
                px = float(r["price"])
            except Exception:
                continue
            by_sym[sym].append((ts, px))
    # 정렬 및 인덱스화
    for sym in list(by_sym.keys()):
        by_sym[sym].sort(key=lambda x: x[0])
    return by_sym

def nearest_price_after(by_sym, sym, t, window):
    if sym not in by_sym: return None, None
    arr = by_sym[sym]
    times = [x[0] for x in arr]
    i = bisect_left(times, t)
    # 1) t 이후 첫 틱
    if i < len(arr):
        ts, px = arr[i]
        if ts - t <= window: return ts, px
    # 2) t 이전 마지막 틱(실패 시 fallback)
    if i > 0:
        ts, px = arr[i-1]
        if t - ts <= window: return ts, px
    return None, None

def load_universe(path):
    if not path or not os.path.exists(path): return None
    out=set()
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s and not s.startswith("#"): out.add(s.upper())
    return out or None

def load_signals(signals_csv, events, sides, universe):
    # 기대 포맷(헤더 없거나 있어도 0열이 ISO일 것):
    # ts,event,side,level,touches,symbol,timeframe,extra,host,message,(optional distance)
    sigs=[]
    with open(signals_csv,"r",encoding="utf-8") as f:
        rows=list(csv.reader(f))
    for i,row in enumerate(rows):
        if not row or len(row)<6: continue
        if i==0 and not row[0].startswith("202"):  # 헤더로 보이면 스킵
            continue
        try:
            ts = parse_dt(row[0])
            event = row[1].strip()
            side  = row[2].strip()
            sym   = row[5].strip().upper()
        except Exception:
            continue
        if events and event not in events: continue
        if sides and side not in sides: continue
        if universe and sym not in universe: continue
        sigs.append((ts,event,side,sym))
    sigs.sort(key=lambda x: x[0])
    return sigs

def net_pnl(entry, exit_, fee):
    # 왕복 수수료 fee (예: 0.001 = 0.1%, 편도 0.05%)
    f_in  = fee/2.0
    f_out = fee/2.0
    return (exit_*(1.0-f_out)/(entry*(1.0+f_in)))-1.0

def main():
    ap = argparse.ArgumentParser(description="Support/Resistance replay backtest (prices.csv 기반)")
    ap.add_argument("--signals-csv", default="./logs/signals_tv.csv")
    ap.add_argument("--prices-csv", default="./logs/prices.csv")
    ap.add_argument("--universe", default="./configs/universe.txt")
    ap.add_argument("--events", default="line_breakout,box_breakout")
    ap.add_argument("--sides", default="support")  # support / resistance / both(comma)
    ap.add_argument("--expiry-min", type=int, default=30)
    ap.add_argument("--price-window-sec", type=int, default=600)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--out", default="./logs/backtest/trades_support.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    events = {s.strip() for s in args.events.split(",") if s.strip()}
    sides  = {s.strip() for s in args.sides.split(",") if s.strip()}
    if "both" in sides: sides = {"support","resistance"}

    uni = load_universe(args.universe)
    prices = load_prices(args.prices_csv, uni)
    window = timedelta(seconds=args.price_window_sec)
    sigs = load_signals(args.signals_csv, events, sides, uni)

    rows=[]
    for ts,event,side,sym in sigs:
        # entry
        ent_ts, ent_px = nearest_price_after(prices, sym, ts, window)
        if ent_px is None: 
            # 가격이 전혀 없으면 스킵
            continue
        # exit at expiry
        exp_ts = ts + timedelta(minutes=args.expiry_min)
        ex_ts, ex_px = nearest_price_after(prices, sym, exp_ts, window)
        if ex_px is None:
            # 만기부근 가격이 없으면 스킵(백테스트 보수적)
            continue
        pnl = net_pnl(ent_px, ex_px, args.fee)
        rows.append({
            "opened_at": ts.isoformat(),
            "symbol": sym,
            "event": event,
            "side": side,
            "closed_at": ex_ts.isoformat(),
            "entry_price": f"{ent_px:.6f}",
            "exit_price": f"{ex_px:.6f}",
            "pnl": f"{pnl:.6f}",
            "fee": f"{args.fee:.6f}",
            "entry_src": "csv",
            "exit_src": "csv",
        })

    # 저장
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "opened_at","symbol","event","side","closed_at",
            "entry_price","exit_price","pnl","fee","entry_src","exit_src"
        ])
        w.writeheader()
        w.writerows(rows)

    # 요약
    n=len(rows)
    if n==0:
        print("No trades generated (check filters / windows).")
        return
    wins = sum(1 for r in rows if float(r["pnl"])>0)
    avg  = sum(float(r["pnl"]) for r in rows)/n
    print(f"Trades: {n} | Win%: {wins/n*100:.1f}% | Avg PnL: {avg*100:.2f}%")
    # 심볼 상위
    by_sym=defaultdict(list)
    for r in rows: by_sym[r["symbol"]].append(float(r["pnl"]))
    top = sorted(((s,sum(v)/len(v),len(v)) for s,v in by_sym.items()), key=lambda x:x[1], reverse=True)[:10]
    print("Top symbols (avg pnl, n):")
    for s,avgp,cnt in top:
        print(f"  {s}: {avgp*100:.2f}% ({cnt})")

if __name__=="__main__":
    main()