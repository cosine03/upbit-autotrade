# paper_trader.py
# Paper Trading Engine skeleton with clean run/stop controls.
# Usage:
#   python paper_trader.py --once
#   python paper_trader.py --run-for 30 --interval 15
#   python paper_trader.py               (forever loop; Ctrl+C to stop)

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

# ---------------------------
# Utilities
# ---------------------------

def now_utc() -> pd.Timestamp:
    # tz-aware UTC timestamp (no tz_localize on aware objects!)
    return pd.Timestamp.now(tz="UTC")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def csv_exists(fp: Path) -> bool:
    return fp.exists() and fp.stat().st_size > 0

def append_csv(fp: Path, df: pd.DataFrame) -> None:
    header = not csv_exists(fp)
    df.to_csv(fp, index=False, mode="a", header=header, encoding="utf-8")

# ---------------------------
# Engine
# ---------------------------

class PaperEngine:
    def __init__(
        self,
        root: Path,
        data_dir: Path,
        logs_dir: Path,
        equity_fp: Path,
        orders_fp: Path,
        fills_fp: Path,
        positions_fp: Path,
        state_fp: Path,
        logger: logging.Logger,
    ):
        self.root = root
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        self.fp_equity = equity_fp
        self.fp_orders = orders_fp
        self.fp_fills = fills_fp
        self.fp_positions = positions_fp
        self.fp_state = state_fp
        self.log = logger

        # State
        self.equity = 1.0  # start at 1.0 (100%)
        self.positions: Dict[str, Dict] = {}
        self.open_orders: List[Dict] = []

        # Init files
        self._init_files()

        self.log.info("PaperEngine initialized")

    def _init_files(self):
        ensure_dir(self.logs_dir)
        ensure_dir(self.data_dir)

        # equity
        if not csv_exists(self.fp_equity):
            pd.DataFrame([{"ts": now_utc().isoformat(), "equity": self.equity}]).to_csv(
                self.fp_equity, index=False, encoding="utf-8"
            )

        # orders
        if not csv_exists(self.fp_orders):
            pd.DataFrame(
                columns=[
                    "ts", "symbol", "side", "qty", "price",
                    "type", "status", "client_id"
                ]
            ).to_csv(self.fp_orders, index=False, encoding="utf-8")

        # fills
        if not csv_exists(self.fp_fills):
            pd.DataFrame(
                columns=[
                    "ts", "symbol", "side", "qty", "price",
                    "fee", "pnl", "order_client_id", "fill_id"
                ]
            ).to_csv(self.fp_fills, index=False, encoding="utf-8")

        # positions
        if not csv_exists(self.fp_positions):
            pd.DataFrame(
                columns=[
                    "ts", "symbol", "qty", "avg_price", "unrealized_pnl"
                ]
            ).to_csv(self.fp_positions, index=False, encoding="utf-8")

        # state json
        if not self.fp_state.exists():
            self._save_state()

    def _save_state(self):
        state = {
            "ts": now_utc().isoformat(),
            "equity": self.equity,
            "positions": self.positions,
            "open_orders": self.open_orders,
        }
        self.fp_state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------
    # One loop body
    # ---------------------------
    def step(self):
        """
        Single iteration:
         1) ingest new signals (TODO: wire to your actual signals)
         2) simulate entries/exits (TODO)
         3) mark-to-market & equity snapshot
         4) persist snapshots
        """
        ts = now_utc()

        # --- (1) ingest signals ---
        # Placeholder: read a signals file if you want; here we only heartbeat.
        # Example to wire later:
        # signals_path = self.root / "logs" / "signals_tv.csv"
        # if signals_path.exists():
        #     sigs = pd.read_csv(signals_path)
        #     # filter/route to strategies...

        # --- (2) simulate orders/fills/positions ---
        # Placeholder NO-OP. Hook your strategy logic here.

        # --- (3) mark-to-market & equity snapshot ---
        # For skeleton: keep equity steady. (You can compute equity from positions here.)
        snap = pd.DataFrame([{"ts": ts.isoformat(), "equity": self.equity}])
        append_csv(self.fp_equity, snap)

        # Positions snapshot (still empty unless you update self.positions)
        pos_rows = []
        for sym, pos in self.positions.items():
            pos_rows.append({
                "ts": ts.isoformat(),
                "symbol": sym,
                "qty": pos.get("qty", 0.0),
                "avg_price": pos.get("avg_price", 0.0),
                "unrealized_pnl": pos.get("unrealized_pnl", 0.0),
            })
        if pos_rows:
            append_csv(self.fp_positions, pd.DataFrame(pos_rows))

        # persist engine state
        self._save_state()

        self.log.debug("step done @ %s", ts.isoformat())

    # ---------------------------
    # Cleanup
    # ---------------------------
    def close(self):
        # If you add pools/threads/files, close them here.
        self.log.info("engine closed")

# ---------------------------
# Logging setup
# ---------------------------

def setup_logger(log_dir: Path) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger("paper")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_dir / "engine.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # avoid duplicate handlers on re-run
    logger.propagate = False
    return logger

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Paper Trading Engine")
    ap.add_argument("--root", type=str, default=".",
                    help="Project root (default=.)")
    ap.add_argument("--once", action="store_true",
                    help="Run single iteration and exit")
    ap.add_argument("--run-for", type=int, default=0,
                    help="Run for N minutes then exit (0=forever)")
    ap.add_argument("--interval", type=float, default=15.0,
                    help="Loop interval seconds (default=15s)")
    ap.add_argument("--expiries", type=str, default="0.5,1,2",
                    help="Expiry hours list, e.g. '0.5,1,2'")
    ap.add_argument("--quick-expiry-secs", type=int, default=None,
                    help="Override all expiries with a short test expiry (seconds), e.g. 36")
    args = ap.parse_args()

    # --- expiries 결정 ---
    if args.quick_expiry_secs is not None and args.quick_expiry_secs > 0:
        expiries_h = [args.quick_expiry_secs / 3600.0]   # 36초 → 0.01h
        quick_note = f"(quick-expiry={args.quick_expiry_secs}s)"
    else:
        expiries_h = [float(x) for x in str(args.expiries).split(",") if x.strip()]
        quick_note = ""

    logger.info(f"expiries_h = {expiries_h} {quick_note}")

    root = Path(args.root).resolve()
    logs_dir = root / "logs" / "paper"
    data_dir = root / "data" / "paper"
    ensure_dir(logs_dir)
    ensure_dir(data_dir)

    logger = setup_logger(logs_dir)
    logger.info("=== Paper Engine START === root=%s", str(root))

    # file paths
    fp_equity    = data_dir / "paper_equity.csv"
    fp_orders    = data_dir / "paper_orders.csv"
    fp_fills     = data_dir / "paper_fills.csv"
    fp_positions = data_dir / "paper_positions.csv"
    fp_state     = data_dir / "engine_state.json"

    eng = PaperEngine(
        root=root_dir,
        expiries_h=expiries_h,
        data_dir=data_dir,
        logs_dir=logs_dir,
        equity_fp=fp_equity,
        orders_fp=fp_orders,
        fills_fp=fp_fills,
        positions_fp=fp_positions,
        state_fp=fp_state,
        logger=logger,
    )

    try:
        if args.once:
            eng.step()
            logger.info("single step done; exiting")
            return

        stop_at: Optional[pd.Timestamp] = None
        if args.run_for > 0:
            stop_at = now_utc() + pd.Timedelta(minutes=args.run_for)
            logger.info("run-for: %d min (stop_at=%s)", args.run_for, stop_at.isoformat())

        while True:
            eng.step()
            if stop_at and now_utc() >= stop_at:
                logger.info("timebox reached; exiting")
                break
            time.sleep(max(0.0, args.interval))

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt - shutting down...")
    finally:
        eng.close()
        logger.info("=== Paper Engine END ===")

if __name__ == "__main__":
    main()