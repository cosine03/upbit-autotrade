#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import logging
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import random
from typing import Optional, List

# ------------------------- utils -------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def ts_str(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ------------------------- data models -------------------------

@dataclass
class OpenTrade:
    id: str
    strategy: str
    expiry_h: float
    side: str
    entry_ts: str
    entry_price: float
    expiry_sec: int

@dataclass
class ClosedTrade:
    id: str
    strategy: str
    expiry_h: float
    side: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    pnl: float
    status: str  # "expired"

# ------------------------- engine -------------------------

class PaperEngine:
    def __init__(self, root: str, expiries_h: List[float], quick_expiry_secs: Optional[int] = None, interval:int=10):
        self.root = root
        self.expiries_h = expiries_h
        self.quick_expiry_secs = quick_expiry_secs
        self.interval = interval
        self.running = True

        self.logdir = os.path.join(root, "logs", "paper")
        ensure_dir(self.logdir)

        self.fp_trades = os.path.join(self.logdir, "trades.csv")
        self.fp_open = os.path.join(self.logdir, "trades_open.csv")
        self.fp_equity = os.path.join(self.logdir, "equity.csv")

        if not os.path.exists(self.fp_equity):
            pd.DataFrame([{"ts": ts_str(now_utc()), "equity": 1.0}]).to_csv(self.fp_equity, index=False)

        # CSV 헤더 보장
        if not os.path.exists(self.fp_trades):
            pd.DataFrame(columns=[
                "id","strategy","expiry_h","side",
                "entry_ts","exit_ts","entry_price","exit_price","pnl","status"
            ]).to_csv(self.fp_trades, index=False)
        if not os.path.exists(self.fp_open):
            pd.DataFrame(columns=[
                "id","strategy","expiry_h","side","entry_ts","entry_price","expiry_sec"
            ]).to_csv(self.fp_open, index=False)

        logging.info(f"PaperEngine initialized expiries={self.expiries_h}, quick={self.quick_expiry_secs}")

    # ---------- IO helpers ----------
    def _read_open(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.fp_open)
        except Exception:
            return pd.DataFrame(columns=[
                "id","strategy","expiry_h","side","entry_ts","entry_price","expiry_sec"
            ])

    def _write_open(self, df: pd.DataFrame):
        df.to_csv(self.fp_open, index=False)

    def _append_closed(self, rows: List[ClosedTrade]):
        if not rows:
            return
        df = pd.DataFrame([asdict(r) for r in rows])
        df.to_csv(self.fp_trades, mode="a", header=not os.path.getsize(self.fp_trades), index=False)

    # ---------- test signal (for EXIT verification) ----------
    def _maybe_open_one_per_expiry(self):
        """
        테스트용: 각 expiry_h마다 열린 포지션이 없으면 1개 생성.
        """
        open_df = self._read_open()
        now_s = ts_str(now_utc())

        created = 0
        for eh in self.expiries_h:
            # 해당 만기의 오픈 포지션 존재 여부
            exists = False
            if not open_df.empty:
                exists = any(abs(open_df["expiry_h"] - eh) < 1e-9)

            if exists:
                continue

            # 새 엔트리 생성
            trade_id = f"T{int(time.time()*1000)}_{str(eh).replace('.','_')}"
            side = random.choice(["long", "short"])
            entry_price = 100.0  # 더미
            expiry_sec = int(self.quick_expiry_secs if self.quick_expiry_secs else eh * 3600)

            row = OpenTrade(
                id=trade_id,
                strategy="test_strategy",
                expiry_h=eh,
                side=side,
                entry_ts=now_s,
                entry_price=entry_price,
                expiry_sec=expiry_sec,
            )
            new_row = pd.DataFrame([asdict(row)])
            frames = [df for df in (open_df, new_row) if not df.empty]
            open_df = pd.concat(frames, ignore_index=True) if frames else new_row
            created += 1

        if created:
            self._write_open(open_df)
            logging.info(f"opened {created} test trade(s)")

    def _close_expired(self):
        """
        만기시간 지난 포지션을 EXIT로 기록하고 open에서 제거.
        """
        open_df = self._read_open()
        if open_df.empty:
            return

        now_dt = now_utc()
        remaining = []
        closed_rows: List[ClosedTrade] = []

        for _, r in open_df.iterrows():
            entry_dt = datetime.strptime(r["entry_ts"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            expiry_td = timedelta(seconds=int(r["expiry_sec"]))
            if now_dt >= entry_dt + expiry_td:
                # 더미 PnL: ±0.1~0.3%
                drift = random.uniform(0.001, 0.003)
                sign = 1 if random.random() < 0.55 else -1
                pnl = sign * drift
                exit_price = float(r["entry_price"]) * (1.0 + pnl)

                closed = ClosedTrade(
                    id=r["id"],
                    strategy=r["strategy"],
                    expiry_h=float(r["expiry_h"]),
                    side=r["side"],
                    entry_ts=r["entry_ts"],
                    exit_ts=ts_str(now_dt),
                    entry_price=float(r["entry_price"]),
                    exit_price=exit_price,
                    pnl=pnl,
                    status="expired",
                )
                closed_rows.append(closed)
            else:
                remaining.append(r)

        # 기록 갱신
        if closed_rows:
            self._append_closed(closed_rows)
            logging.info(f"closed {len(closed_rows)} trade(s)")

        if remaining:
            self._write_open(pd.DataFrame(remaining))
        else:
            # 모두 닫혔으면 빈 프레임으로 초기화
            self._write_open(pd.DataFrame(columns=[
                "id","strategy","expiry_h","side","entry_ts","entry_price","expiry_sec"
            ]))

    # ---------- main ticks ----------
    def run_once(self):
        # 1) 테스트용 entry 생성(만기별 1개씩)
        self._maybe_open_one_per_expiry()
        # 2) 만기 도달한 포지션 종료 → trades.csv에 EXIT append
        self._close_expired()

    def loop(self, interval_sec: int, stop_at: Optional[datetime] = None):
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

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="Run one tick and exit")
    p.add_argument("--run-for", type=int, default=None, help="Run for N minutes")
    p.add_argument("--interval", type=int, default=10, help="Loop interval seconds")
    p.add_argument("--expiries", type=str, default="0.5,1,2", help="Expiry hours list, e.g. '0.5,1,2'")
    p.add_argument("--quick-expiry-secs", type=int, default=None, help="Quick test expiry seconds (e.g. 36)")
    return p.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("=== Paper Engine START ===")

    args = parse_args()
    root = os.getcwd()

    if args.quick_expiry_secs and args.quick_expiry_secs > 0:
        expiries_h = [args.quick_expiry_secs / 3600.0]
        note = f"(quick={args.quick_expiry_secs}s)"
    else:
        expiries_h = [float(x) for x in str(args.expiries).split(",") if x.strip()]
        note = ""

    logging.info(f"expiries_h={expiries_h} {note}")

    eng = PaperEngine(root=root, expiries_h=expiries_h,
                      quick_expiry_secs=args.quick_expiry_secs,
                      interval=args.interval)

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