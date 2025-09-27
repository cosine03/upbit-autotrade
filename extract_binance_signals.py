
import argparse
import sys
import pandas as pd

def is_usdt_symbol(sym: str) -> bool:
    return isinstance(sym, str) and sym.endswith("USDT") and len(sym) > 5  # e.g., BTCUSDT

def map_usdt_to_upbit(sym: str) -> str:
    # BTCUSDT -> KRW-BTC
    if is_usdt_symbol(sym):
        base = sym[:-4]  # strip 'USDT'
        return f"KRW-{base}"
    return sym

def main():
    ap = argparse.ArgumentParser(description="Extract Binance(USDT) signals and map to Upbit symbols optionally.")
    ap.add_argument("--src", required=True, help="Path to signals_tv.csv (raw webhook sink).")
    ap.add_argument("--out-raw", required=True, help="Output path for Binance-only raw signals (BTCUSDT etc.).")
    ap.add_argument("--out-upbit", required=True, help="Output path for Binance signals mapped to Upbit tickers (KRW-XXX).")
    ap.add_argument("--side", default="any", choices=["any","resistance","support"], help="Filter side if needed.")
    ap.add_argument("--min-touches", type=int, default=0, help="Minimum touches filter (0 to skip).")
    ap.add_argument("--events", default="", help="Comma-separated events to include (empty for all). e.g. 'price_in_box,box_breakout,line_breakout'")
    args = ap.parse_args()

    # Load with auto sep detection
    try:
        df = pd.read_csv(args.src, sep=None, engine="python")
    except Exception as e:
        print(f"[ERR] failed to read {args.src}: {e}", file=sys.stderr)
        sys.exit(2)

    # Normalize required columns existence
    required = ["ts","event","side","level","touches","symbol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERR] missing columns in source: {missing}", file=sys.stderr)
        print(f"[HINT] columns found: {list(df.columns)}", file=sys.stderr)
        sys.exit(3)

    # Basic cleaning
    df["symbol"] = df["symbol"].astype(str).str.strip()

    # Side filter
    if args.side != "any":
        df = df[df["side"] == args.side]

    # touches filter
    if args.min_touches and "touches" in df.columns:
        df = df[pd.to_numeric(df["touches"], errors="coerce").fillna(-1) >= args.min_touches]

    # events filter
    if args.events:
        allow = [e.strip() for e in args.events.split(",") if e.strip()]
        if allow:
            df = df[df["event"].isin(allow)]

    # 1) Binance-only raw
    df_bin = df[df["symbol"].apply(is_usdt_symbol)].copy()
    df_bin.to_csv(args.out_raw, index=False, encoding="utf-8")
    print(f"[OUT] saved raw Binance-only -> {args.out_raw} (rows={len(df_bin)})")

    # 2) Binance mapped to Upbit symbols (for Upbit price backtests)
    if not df_bin.empty:
        df_map = df_bin.copy()
        df_map["symbol"] = df_map["symbol"].map(map_usdt_to_upbit)
    else:
        df_map = df_bin.copy()
    df_map.to_csv(args.out_upbit, index=False, encoding="utf-8")
    print(f"[OUT] saved mapped for Upbit price -> {args.out_upbit} (rows={len(df_map)})")

    # Quick summaries
    if not df_bin.empty:
        by_event = df_bin["event"].value_counts(dropna=False).to_dict()
        by_side  = df_bin["side"].value_counts(dropna=False).to_dict()
        print(f"[SUM] Binance raw: events={by_event}")
        print(f"[SUM] Binance raw: side={by_side}")
    else:
        print("[SUM] Binance raw: (empty)")

if __name__ == "__main__":
    main()
