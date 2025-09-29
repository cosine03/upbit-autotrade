#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
paper_trader.py
- Paper trading engine for strategy signal simulation
- Supports custom expiries (hours) and quick-expiry mode (seconds, for testing EXIT quickly)
"""

import os
import sys
import time
import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def now_utc():
    return datetime.now(timezone.utc)

def ts_to_str(ts: datetime):
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------

class PaperEngine:
    def __init__(self, root: str, expiries_h: list[float]):
        self.root = root
        self.expiries_h = expiries_h
        self.running = True

        # logs dir
        self.logdir = os.path.join(root, "logs", "paper")
        os.makedirs(self.logdir, exist_ok=True)

        # trades file
        self.fp_trades = os.path.join(self.logdir, "trades.csv")
        self.fp_equity = os.path.join(self.logdir, "equity.csv")

        # init equity
        if not os.path.exists(self.fp_equity):
            pd.DataFrame([{"ts": now_utc(), "equity": 1.0}]).to_csv(self.fp_equity, index=False)

        logging.info(f"PaperEngine initialized, expiries_h={self.expiries_h}")

    def run_once(self):
        """
        Run a single tick of the engine:
        - Check signals
        - Open new trades
        - Check expiries → close trades
        """
        now = now_utc()
        logging.debug(f"tick {ts_to_str(now)}")

        # [여기에 entry/exit 로직 구현]
        # 예시: 만기 조건 체크
        # for each open trade:
        #   if now >= trade['entry_ts'] + expiry_td → close trade, write exit

        pass

    def loop(self, interval_sec: int, stop_at: datetime | None = None):
        logging.info("engine loop started")
        try:
            while self.running:
                self.run_once()
                if stop_at and now_utc() >= stop_at:
                    logging.info("timebox reached; exiting")
                    break
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt - shutting down...")
        finally:
            self.close()

    def close(self):
        self.running = False
        logging.info("engine closed")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--run-for", type=int, default=None, help="Run for N minutes")
    parser.add_argument("--interval", type=int, default=10, help="Loop interval seconds")
    parser.add_argument("--expiries", type=str, default="0.5,1,2",
                        help="Expiry hours list, e.g. '0.5,1,2'")
    parser.add_argument("--quick-expiry-secs", type=int, default=None,
                        help="Override expiries with a quick test expiry in seconds (e.g. 36)")
    return parser.parse_args()

def main():
    args = parse_args()

    # logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("=== Paper Engine START ===")

    root_dir = os.getcwd()

    # --- expiries 결정 ---
    if args.quick_expiry_secs is not None and args.quick_expiry_secs > 0:
        expiries_h = [args.quick_expiry_secs / 3600.0]   # 초를 시간 단위로 변환
        quick_note = f"(quick-expiry={args.quick_expiry_secs}s)"
    else:
        expiries_h = [float(x) for x in str(args.expiries).split(",") if x.strip()]
        quick_note = ""

    logging.info(f"expiries_h = {expiries_h} {quick_note}")

    # 엔진 생성
    eng = PaperEngine(root=root_dir, expiries_h=expiries_h)

    # 실행 모드
    if args.once:
        eng.run_once()
        eng.close()
        logging.info("single step done; exiting")
    elif args.run_for:
        stop_at = now_utc() + timedelta(minutes=args.run_for)
        logging.info(f"run-for: {args.run_for} min (stop_at={stop_at.isoformat()})")
        eng.loop(interval_sec=args.interval, stop_at=stop_at)
    else:
        eng.loop(interval_sec=args.interval)

    logging.info("=== Paper Engine END ===")

if __name__ == "__main__":
    main()