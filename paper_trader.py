#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import logging
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

import pandas as pd

# =========================================================
# Utils
# =========================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def ts_str(ts: datetime) -> str:
    # 표준화(UTC, 초단위)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# =========================================================
# CSV Schema (고정)
# =========================================================
SCHEMA_OPEN = [
    "id","strategy","symbol","event","expiry_h","side",
    "entry_ts","entry_price","expiry_sec","distance_pct","run_id"
]
SCHEMA_CLOSED = [
    "id","strategy","symbol","event","expiry_h","side",
    "entry_ts","exit_ts","entry_price","exit_price","pnl","status",
    "distance_pct","run_id","fee"
]

def reindex_schema(df: pd.DataFrame, schema: List[str]) -> pd.DataFrame:
    """DataFrame을 schema 순서로 재정렬하고 누락컬럼을 채운다."""
    if df is None:
        return pd.DataFrame(columns=schema)
    for col in schema:
        if col not in df.columns:
            df[col] = pd.NA
    # 초과 컬럼은 뒤로 보존(디버그용)
    extra = [c for c in df.columns if c not in schema]
    return df[schema + extra]

# =========================================================
# Data Models
# =========================================================
@dataclass
class OpenTrade:
    id: str
    strategy: str
    symbol: str
    event: str
    expiry_h: float
    side: str
    entry_ts: str
    entry_price: float
    expiry_sec: int
    distance_pct: Optional[float] = None
    run_id: Optional[str] = None

@dataclass
class ClosedTrade:
    id: str
    strategy: str
    symbol: str
    event: str
    expiry_h: float
    side: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    pnl: float          # net pnl (fee 반영)
    status: str         # "expired"
    distance_pct: Optional[float] = None
    run_id: Optional[str] = None
    fee: Optional[float] = None

# =========================================================
# Engine
# =========================================================
class PaperEngine:
    def __init__(
        self,
        root: str,
        expiries_h: List[float],
        interval: int = 10,
        quick_expiry_secs: Optional[int] = None,
        # signal options
        signals_csv: Optional[str] = None,
        long_only: bool = False,
        signals_recent_sec: int = 3600,
        min_distance_pct: Optional[float] = None,
        max_distance_pct: Optional[float] = None,
        allow_events: Optional[List[str]] = None,
        cooldown_sec: int = 0,
        max_opens_per_tick: int = 999999,
        max_open_positions: int = 999999,
        fee: float = 0.0,
    ):
        self.root = root
        self.expiries_h = expiries_h
        self.interval = interval
        self.quick_expiry_secs = quick_expiry_secs

        # signals
        self.signals_csv = signals_csv
        self.long_only = long_only
        self.signals_recent_sec = signals_recent_sec
        self.min_distance_pct = min_distance_pct
        self.max_distance_pct = max_distance_pct
        self.allow_events = [e.strip() for e in (allow_events or []) if str(e).strip()]
        self.cooldown_sec = cooldown_sec
        self.max_opens_per_tick = max_opens_per_tick
        self.max_open_positions = max_open_positions
        self.fee = fee

        # paths
        self.logdir = os.path.join(root, "logs", "paper")
        ensure_dir(self.logdir)
        self.fp_trades = os.path.join(self.logdir, "trades.csv")
        self.fp_open = os.path.join(self.logdir, "trades_open.csv")
        self.fp_equity = os.path.join(self.logdir, "equity.csv")

        # equity seed
        if not os.path.exists(self.fp_equity):
            pd.DataFrame([{"ts": ts_str(now_utc()), "equity": 1.0}]).to_csv(self.fp_equity, index=False)

        # file headers 보장
        if not os.path.exists(self.fp_open):
            pd.DataFrame(columns=SCHEMA_OPEN).to_csv(self.fp_open, index=False)
        if not os.path.exists(self.fp_trades):
            pd.DataFrame(columns=SCHEMA_CLOSED).to_csv(self.fp_trades, index=False)

        # cooldown memory
        self._last_open_at: Dict[str, datetime] = {}

        logging.info(
            "PaperEngine initialized expiries=%s, long_only=%s, signals_csv=%s, recent=%ss, "
            "dist=(%s,%s), allow_events=%s, cooldown=%ss, max_per_tick=%s, max_open_positions=%s, "
            "fee=%s, quick=%s",
            self.expiries_h, self.long_only, self.signals_csv, self.signals_recent_sec,
            self.min_distance_pct, self.max_distance_pct, self.allow_events, self.cooldown_sec,
            self.max_opens_per_tick, self.max_open_positions, self.fee, self.quick_expiry_secs
        )

    # ---------- Admin ----------
    def reset_open_book(self):
        """미결 오픈북 초기화(기존 파일은 archive)."""
        try:
            if os.path.exists(self.fp_open):
                arch = os.path.join(self.logdir, f"trades_open_archive_{int(time.time())}.csv")
                os.replace(self.fp_open, arch)
        finally:
            pd.DataFrame(columns=SCHEMA_OPEN).to_csv(self.fp_open, index=False)
        logging.info("open book reset (archived if existed)")

    # ---------- IO helpers ----------
    def _read_open(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.fp_open)
        except Exception:
            df = pd.DataFrame(columns=SCHEMA_OPEN)
        return reindex_schema(df, SCHEMA_OPEN)

    def _write_open(self, df: pd.DataFrame):
        reindex_schema(df, SCHEMA_OPEN).to_csv(self.fp_open, index=False)

    def _append_closed(self, rows: List[ClosedTrade]):
        if not rows:
            return 0
        df_new = pd.DataFrame([asdict(r) for r in rows])
        df_new = reindex_schema(df_new, SCHEMA_CLOSED)

        header_flag = not (os.path.exists(self.fp_trades) and os.path.getsize(self.fp_trades) > 0)
        # 기존 파일의 헤더 순서를 따르도록 시도(혼선 방지)
        if not header_flag:
            try:
                exist_cols = list(pd.read_csv(self.fp_trades, nrows=0).columns)
                for col in exist_cols:
                    if col not in df_new.columns:
                        df_new[col] = pd.NA
                df_new = df_new[exist_cols]
            except Exception:
                header_flag = True

        df_new.to_csv(self.fp_trades, mode="a", header=header_flag, index=False)
        return len(rows)

    # =====================================================
    # Signals → Entries
    # =====================================================
    def _load_signal_df(self) -> pd.DataFrame:
        if not self.signals_csv or not os.path.exists(self.signals_csv):
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.signals_csv)
        except Exception:
            return pd.DataFrame()

        # 컬럼 표준화 시도
        # 기대 컬럼: ts, symbol, event, side, distance_pct
        rename_map = {}
        # 타임스탬프 후보
        for cand in ["ts", "signal_ts", "time", "timestamp"]:
            if cand in df.columns:
                rename_map[cand] = "ts"
                break
        # 심볼 후보
        for cand in ["symbol", "ticker", "market", "pair"]:
            if cand in df.columns:
                rename_map[cand] = "symbol"
                break
        # 이벤트 후보
        for cand in ["event", "evt", "signal", "type"]:
            if cand in df.columns:
                rename_map[cand] = "event"
                break
        # 방향 후보
        for cand in ["side", "direction"]:
            if cand in df.columns:
                rename_map[cand] = "side"
                break
        # distance 후보
        for cand in ["distance_pct", "distance", "price_in_box", "dist"]:
            if cand in df.columns:
                rename_map[cand] = "distance_pct"
                break

        if rename_map:
            df = df.rename(columns=rename_map)

        # 필수 컬럼 보정
        for need in ["ts", "symbol", "event"]:
            if need not in df.columns:
                df[need] = pd.NA
        if "side" not in df.columns:
            df["side"] = "long"  # 기본 long
        if "distance_pct" not in df.columns:
            df["distance_pct"] = pd.NA

        # 타입 변환/정리
        # ts를 datetime으로
        def _parse_ts(x):
            if pd.isna(x):
                return pd.NaT
            s = str(x)
            try:
                # 2025-09-29 05:50:59 or ISO
                return pd.to_datetime(s, utc=True)
            except Exception:
                # epoch(sec or ms)
                try:
                    xv = float(s)
                    if xv > 1e12:  # ms
                        xv = xv / 1000.0
                    return pd.to_datetime(xv, unit="s", utc=True)
                except Exception:
                    return pd.NaT

        df["ts"] = df["ts"].apply(_parse_ts)

        # distance float
        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return float("nan")

        df["distance_pct"] = df["distance_pct"].apply(_to_float)

        return df

    def _open_from_signals(self) -> int:
        """CSV 시그널에서 신규 엔트리 생성."""
        df = self._load_signal_df()
        if df.empty:
            return 0

        # 최근 N초 필터
        if self.signals_recent_sec and self.signals_recent_sec > 0:
            cutoff = now_utc() - timedelta(seconds=self.signals_recent_sec)
            df = df[df["ts"] >= pd.Timestamp(cutoff)]
        # 이벤트 필터
        if self.allow_events:
            df = df[df["event"].astype(str).isin(self.allow_events)]
        # 방향 필터
        if self.long_only:
            df = df[df["side"].astype(str).str.lower() == "long"]
        # distance 범위
        if self.min_distance_pct is not None:
            df = df[df["distance_pct"] >= self.min_distance_pct]
        if self.max_distance_pct is not None:
            df = df[df["distance_pct"] <= self.max_distance_pct]

        if df.empty:
            return 0

        # 오픈북/쿨다운/최대치 체크
        open_df = self._read_open()
        now_s = ts_str(now_utc())
        opened = 0

        # 현재 오픈 포지션 수 제한
        open_now = 0 if open_df.empty else len(open_df.index)
        if open_now >= self.max_open_positions:
            return 0

        # expiry 설정(첫 만기 사용; quick이면 quick)
        if self.quick_expiry_secs and self.quick_expiry_secs > 0:
            exp_sec = int(self.quick_expiry_secs)
            exp_h = self.quick_expiry_secs / 3600.0
        else:
            exp_h = float(self.expiries_h[0])
            exp_sec = int(exp_h * 3600)

        for _, r in df.sort_values("ts").iterrows():
            if opened >= self.max_opens_per_tick:
                break
            if open_now + opened >= self.max_open_positions:
                break

            symbol = str(r.get("symbol", "")).strip()
            event = str(r.get("event", "")).strip()
            side = str(r.get("side", "long")).strip().lower()
            dist = r.get("distance_pct", float("nan"))

            # cooldown
            if self.cooldown_sec > 0 and symbol:
                last = self._last_open_at.get(symbol)
                if last and (now_utc() - last).total_seconds() < self.cooldown_sec:
                    continue

            # 같은 심볼/만기 중복 방지: 이미 열려있는지 체크
            duplicate = False
            if not open_df.empty:
                dup_mask = (open_df["symbol"].astype(str) == symbol) & (open_df["expiry_h"].astype(float) == float(exp_h))
                if dup_mask.any():
                    duplicate = True
            if duplicate:
                continue

            trade_id = f"S{int(time.time()*1000)}_{symbol}_{event}_{str(exp_h).replace('.','_')}"

            row = OpenTrade(
                id=trade_id,
                strategy="tv_signal",
                symbol=symbol,
                event=event,
                expiry_h=exp_h,
                side=side,
                entry_ts=now_s,
                entry_price=100.0,        # 더미 가격(실거래 연동 전)
                expiry_sec=exp_sec,
                distance_pct=None if pd.isna(dist) else float(dist),
                run_id=None
            )
            new_row = pd.DataFrame([asdict(row)])
            open_df = pd.concat([open_df, reindex_schema(new_row, SCHEMA_OPEN)], ignore_index=True)
            opened += 1
            if symbol:
                self._last_open_at[symbol] = now_utc()

        if opened:
            self._write_open(open_df)

        if opened:
            logging.info("opened %d trade(s) from signals", opened)
        return opened

    # =====================================================
    # Dummy test entries (만기 검증용)
    # =====================================================
    def _maybe_open_one_per_expiry(self) -> int:
        """테스트용: 만기별로 미결 없으면 하나씩 오픈."""
        open_df = self._read_open()
        now_s = ts_str(now_utc())
        created = 0
        for eh in self.expiries_h:
            # 이미 같은 만기의 미결이 있으면 skip
            exists = False
            if not open_df.empty:
                try:
                    exists = (open_df["expiry_h"].astype(float) == float(eh)).any()
                except Exception:
                    exists = False
            if exists:
                continue

            trade_id = f"T{int(time.time()*1000)}_{str(eh).replace('.','_')}"
            side = "long" if self.long_only else random.choice(["long", "short"])
            entry_price = 100.0
            expiry_sec = int(self.quick_expiry_secs if self.quick_expiry_secs else eh * 3600)

            row = OpenTrade(
                id=trade_id,
                strategy="test_strategy",
                symbol="",
                event="",
                expiry_h=float(eh),
                side=side,
                entry_ts=now_s,
                entry_price=entry_price,
                expiry_sec=expiry_sec,
                distance_pct=None,
                run_id=None
            )
            new_row = pd.DataFrame([asdict(row)])
            open_df = pd.concat([open_df, reindex_schema(new_row, SCHEMA_OPEN)], ignore_index=True)
            created += 1

        if created:
            self._write_open(open_df)
            logging.info("opened %d test trade(s)", created)
        return created

    # =====================================================
    # Close expired
    # =====================================================
    def _close_expired(self) -> int:
        open_df = self._read_open()
        if open_df.empty:
            return 0

        now_dt = now_utc()
        remain_rows = []
        closed: List[ClosedTrade] = []

        for _, r in open_df.iterrows():
            try:
                entry_dt = pd.to_datetime(str(r["entry_ts"]), utc=True).to_pydatetime()
            except Exception:
                # 잘못 기록된 행은 남기지 않고 skip
                continue

            eh = float(r.get("expiry_h", 0.0))
            exp_sec = int(r.get("expiry_sec", int(eh * 3600)))
            if now_dt >= entry_dt + timedelta(seconds=exp_sec):
                # 더미 가격 변화 (±0.1~0.3%)
                drift = random.uniform(0.001, 0.003)
                sign = 1 if random.random() < 0.55 else -1
                gross = sign * drift
                # 왕복 수수료 2 * fee
                net = gross - (2 * self.fee)

                entry_price = float(r.get("entry_price", 100.0))
                exit_price = entry_price * (1.0 + net)

                closed.append(ClosedTrade(
                    id=str(r.get("id", "")),
                    strategy=str(r.get("strategy", "")),
                    symbol=str(r.get("symbol", "")),
                    event=str(r.get("event", "")),
                    expiry_h=float(r.get("expiry_h", 0.0)),
                    side=str(r.get("side", "")),
                    entry_ts=str(r.get("entry_ts", "")),
                    exit_ts=ts_str(now_dt),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=net,
                    status="expired",
                    distance_pct=(None if pd.isna(r.get("distance_pct", pd.NA)) else float(r.get("distance_pct"))),
                    run_id=str(r.get("run_id", "")) if r.get("run_id", None) is not None else None,
                    fee=self.fee
                ))
            else:
                remain_rows.append(r)

        if closed:
            n = self._append_closed(closed)
            logging.info("closed %d trade(s)", n)

        # 남은 open 갱신
        if remain_rows:
            self._write_open(pd.DataFrame(remain_rows))
        else:
            self._write_open(pd.DataFrame(columns=SCHEMA_OPEN))

        return len(closed)

    # =====================================================
    # Ticks
    # =====================================================
    def run_once(self) -> None:
        # 1) 만기 도달 청산
        n_closed = self._close_expired()
        # 2) 시그널 기반 오픈(있으면), 없으면 테스트 엔트리
        if self.signals_csv:
            n_opened = self._open_from_signals()
        else:
            n_opened = self._maybe_open_one_per_expiry()
        logging.info("tick summary: opened=%s closed=%s", n_opened, n_closed)

    def loop(self, interval_sec: int, stop_at: Optional[datetime] = None):
        logging.info("engine loop started")
        try:
            while True:
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
        logging.info("engine closed")


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    # run modes
    p.add_argument("--once", action="store_true", help="Run one tick and exit")
    p.add_argument("--run-for", type=int, default=None, help="Run for N minutes")
    p.add_argument("--interval", type=int, default=10, help="Loop interval seconds")

    # expiries
    p.add_argument("--expiries", type=str, default="0.5,1,2", help="Expiry hours list, e.g. '0.5,1,2'")
    p.add_argument("--quick-expiry-secs", type=int, default=None, help="Quick test expiry seconds (e.g. 36)")

    # signals
    p.add_argument("--signals-csv", type=str, default=None, help="Signals CSV path")
    p.add_argument("--long-only", action="store_true", help="Use only long side")
    p.add_argument("--signals-recent-sec", type=int, default=3600, help="Only signals within N seconds")
    p.add_argument("--min-distance-pct", type=float, default=None, help="Min distance filter")
    p.add_argument("--max-distance-pct", type=float, default=None, help="Max distance filter")
    p.add_argument("--allow-events", type=str, default=None, help="Comma separated event whitelist")
    p.add_argument("--cooldown-sec", type=int, default=0, help="Per-symbol cooldown seconds")
    p.add_argument("--max-opens-per-tick", type=int, default=999999, help="Cap #opens per tick")
    p.add_argument("--max-open-positions", type=int, default=999999, help="Cap total open positions")
    p.add_argument("--fee", type=float, default=0.0, help="Per-side fee (net = gross - 2*fee)")

    # housekeeping
    p.add_argument("--reset-open", action="store_true", help="Clear leftover open-book at start")
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

    # expiries 계산
    if args.quick_expiry_secs and args.quick_expiry_secs > 0:
        expiries_h = [args.quick_expiry_secs / 3600.0]
        logging.info("expiries_h=%s (quick=%ss)", expiries_h, args.quick_expiry_secs)
    else:
        expiries_h = [float(x) for x in str(args.expiries).split(",") if x.strip()]
        logging.info("expiries_h=%s", expiries_h)

    # allow events list
    allow_events = None
    if args.allow_events:
        allow_events = [x.strip() for x in str(args.allow_events).split(",") if x.strip()]

    eng = PaperEngine(
        root=root,
        expiries_h=expiries_h,
        interval=args.interval,
        quick_expiry_secs=args.quick_expiry_secs,
        signals_csv=args.signals_csv,
        long_only=args.long_only,
        signals_recent_sec=args.signals_recent_sec,
        min_distance_pct=args.min_distance_pct,
        max_distance_pct=args.max_distance_pct,
        allow_events=allow_events,
        cooldown_sec=args.cooldown_sec,
        max_opens_per_tick=args.max_opens_per_tick,
        max_open_positions=args.max_open_positions,
        fee=args.fee,
    )

    logging.info("long_only=%s", args.long_only)

    if args.reset_open:
        eng.reset_open_book()

    if args.once:
        eng.run_once()
        eng.close()
        logging.info("single step done; exiting")
    elif args.run_for:
        stop_at = now_utc() + timedelta(minutes=args.run_for)
        logging.info("run-for: %s min (stop_at=%s)", args.run_for, stop_at.isoformat())
        eng.loop(interval_sec=args.interval, stop_at=stop_at)
    else:
        eng.loop(interval_sec=args.interval)

    logging.info("=== Paper Engine END ===")

if __name__ == "__main__":
    main()