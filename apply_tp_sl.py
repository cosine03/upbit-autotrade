# apply_tp_sl.py
# -*- coding: utf-8 -*-
import csv, sys, argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict

def parse_dt(s):
    # ts가 "2025-10-01T12:34:56+00:00" 또는 "2025-10-01 12:34:56+00:00" 형태라고 가정
    s = s.replace(" ", "T")
    return datetime.fromisoformat(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-csv", required=True)         # replay_backtest 결과(csv)
    ap.add_argument("--prices-csv", required=True)         # prices_merged.csv
    ap.add_argument("--out", required=True)                # 재계산 결과 csv
    ap.add_argument("--tp", type=float, default=0.015)     # 1.5%
    ap.add_argument("--sl", type=float, default=0.007)     # 0.7%
    ap.add_argument("--time-col", default="ts_open")
    ap.add_argument("--entry-col", default="entry")
    ap.add_argument("--symbol-col", default="symbol")
    ap.add_argument("--side-col", default="side_used")     # long/short가 아닌 경우 long만 사용했다고 가정
    ap.add_argument("--expiry-min-col", default="expiry_min")
    args = ap.parse_args()

    # 1) 가격 인덱스: symbol -> {ts -> last_price}
    # prices_merged.csv: ts,symbol,price,src
    price_map = defaultdict(dict)
    with open(args.prices_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                sym = row["symbol"]
                ts  = parse_dt(row["ts"])
                px  = float(row["price"])
            except Exception:
                continue
            price_map[sym][ts] = px

    # 심볼별 타임스탬프 목록(정렬) 캐시
    sorted_ts = {}
    for sym, m in price_map.items():
        sorted_ts[sym] = sorted(m.keys())

    def window_prices(sym, t0, t1):
        if sym not in sorted_ts:
            return []
        arr = sorted_ts[sym]
        # 이진탐색 대신 선형 but 충분히 빠름(필요시 bisect 가능)
        return [(t, price_map[sym][t]) for t in arr if t0 <= t <= t1]

    trades = []
    with open(args.trades_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        # 결과 필드 추가
        extra_cols = ["tp", "sl", "hit", "hit_time", "pnl_pct_tp_sl"]
        out_fields = fields + [c for c in extra_cols if c not in fields]
        for row in r:
            try:
                sym = row[args.symbol_col]
                t_open = parse_dt(row[args.time_col])
                entry = float(row[args.entry_col])
                expiry_min = float(row.get(args.expiry_min_col, 30))
                side = (row.get(args.side_col) or "support").lower()
            except Exception:
                trades.append(row)
                continue

            # 만기 구간
            t_end = t_open + timedelta(minutes=expiry_min)

            # 목표/손절
            if side == "resistance":
                # 보통 숏 관점이지만 여기선 PIB->반전 롱 가정(과거 스크립트 컨벤션에 맞춰 롱으로 해석)
                # 만약 실제가 다르면 여기서 분기해 주세요.
                tp_px = entry * (1.0 + args.tp)
                sl_px = entry * (1.0 - args.sl)
                is_long = True
            else:
                # support는 롱으로 가정
                tp_px = entry * (1.0 + args.tp)
                sl_px = entry * (1.0 - args.sl)
                is_long = True

            # 가격 경로 탐색 (선터치 우선)
            series = window_prices(sym, t_open, t_end)
            hit = ""
            hit_time = ""
            pnl_pct = None

            if series:
                for (t, px) in series:
                    if is_long:
                        if px >= tp_px:
                            hit = "TP"; hit_time = t.isoformat()
                            pnl_pct = args.tp
                            break
                        if px <= sl_px:
                            hit = "SL"; hit_time = t.isoformat()
                            pnl_pct = -args.sl
                            break
                    else:
                        # 필요시 숏 로직 (여기선 사용 안함)
                        pass

                if not hit:
                    # 만기 시점 가격으로 결과 계산
                    # 가장 마지막가
                    last_t, last_px = series[-1]
                    pnl_pct = (last_px - entry) / entry
                    hit = "EXPIRY"
                    hit_time = last_t.isoformat()
            else:
                # 가격 데이터 없으면 원본 그대로 두되 표시
                hit = "NO_DATA"
                hit_time = ""
                pnl_pct = None

            row["tp"] = f"{args.tp:.4f}"
            row["sl"] = f"{args.sl:.4f}"
            row["hit"] = hit
            row["hit_time"] = hit_time
            row["pnl_pct_tp_sl"] = f"{pnl_pct:.5f}" if pnl_pct is not None else ""

            trades.append(row)

    # 저장
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(trades)

    # 요약
    valid = [t for t in trades if t.get("pnl_pct_tp_sl")]
    n = len(valid)
    if n:
        wins = sum(1 for t in valid if float(t["pnl_pct_tp_sl"]) > 0)
        avg = sum(float(t["pnl_pct_tp_sl"]) for t in valid) / n
        print(f"TP/SL Recalc -> Trades: {n} | Win%: {wins/n*100:.1f}% | Avg PnL: {avg*100:.2f}%")
    else:
        print("TP/SL Recalc -> no valid trades")
if __name__ == "__main__":
    main()