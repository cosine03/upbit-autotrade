#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, argparse, logging, random
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple

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
    symbol: str = ""
    event: str = ""
    signal_ts: str = ""
    distance_pct: Optional[float] = None

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
    symbol: str = ""
    event: str = ""
    signal_ts: str = ""
    distance_pct: Optional[float] = None

# ------------------------- engine -------------------------
class PaperEngine:
    def __init__(
        self,
        root: str,
        expiries_h: List[float],
        quick_expiry_secs: Optional[int] = None,
        interval: int = 10,
        # signals / filters
        signals_csv: Optional[str] = None,
        long_only: bool = False,
        bias_side: str = "resistance",
        signals_recent_sec: int = 180,
        min_distance_pct: Optional[float] = None,
        max_distance_pct: Optional[float] = None,
        allow_events: Optional[List[str]] = None,
        cooldown_sec: int = 300,
        max_opens_per_tick: int = 3,
        max_open_positions: int = 30,
        fee: float = 0.001,
        reset_open: bool = False,
    ):
        self.root = root
        self.expiries_h = expiries_h
        self.quick_expiry_secs = quick_expiry_secs
        self.interval = interval

        self.signals_csv = signals_csv
        self.long_only = long_only
        self.bias_side = bias_side
        self.signals_recent_sec = int(signals_recent_sec)
        self.min_distance_pct = min_distance_pct
        self.max_distance_pct = max_distance_pct
        # 기본: box_breakout, line_breakout + 요청대로 price_in_box도 참고용 포함
        self.allow_events = allow_events or ["box_breakout", "line_breakout", "price_in_box"]
        self.cooldown_sec = int(cooldown_sec)
        self.max_opens_per_tick = int(max_opens_per_tick)
        self.max_open_positions = int(max_open_positions)
        self.fee = float(fee)

        self.running = True
        self.logdir = os.path.join(root, "logs", "paper")
        ensure_dir(self.logdir)
        self.fp_trades = os.path.join(self.logdir, "trades.csv")
        self.fp_open = os.path.join(self.logdir, "trades_open.csv")
        self.fp_equity = os.path.join(self.logdir, "equity.csv")

        if not os.path.exists(self.fp_equity):
            pd.DataFrame([{"ts": ts_str(now_utc()), "equity": 1.0}]).to_csv(self.fp_equity, index=False)

        # CSV 헤더 보장 (폐쇄/오픈 공통 슈퍼셋으로 고정)
        trades_cols = [
            "id","strategy","expiry_h","side",
            "symbol","event","signal_ts","distance_pct",
            "entry_ts","exit_ts","entry_price","exit_price","pnl","status"
        ]
        open_cols = [
            "id","strategy","expiry_h","side",
            "entry_ts","entry_price","expiry_sec",
            "symbol","event","signal_ts","distance_pct"
        ]
        self.trades_cols = trades_cols
        self.open_cols = open_cols

        if not os.path.exists(self.fp_trades):
            pd.DataFrame(columns=trades_cols).to_csv(self.fp_trades, index=False)
        if not os.path.exists(self.fp_open):
            pd.DataFrame(columns=open_cols).to_csv(self.fp_open, index=False)

        if reset_open:
            # 기존 오픈북 아카이브 후 리셋
            if os.path.exists(self.fp_open) and os.path.getsize(self.fp_open) > 0:
                stamp = int(time.time())
                arch = os.path.join(self.logdir, f"trades_open_{stamp}.csv")
                try:
                    os.replace(self.fp_open, arch)
                except Exception:
                    pass
            pd.DataFrame(columns=open_cols).to_csv(self.fp_open, index=False)
            logging.info("open book reset (archived if existed)")

        # 내부 상태
        self.last_open_ts_by_symbol: Dict[str, float] = {}  # cooldown 관리(epoch secs)

        logging.info(
            "PaperEngine initialized expiries=%s, long_only=%s, bias=%s, "
            "signals_csv=%s, recent=%ss, dist=(%s,%s), allow_events=%s, "
            "cooldown=%ss, max_per_tick=%s, max_open_positions=%s, fee=%s, quick=%s",
            self.expiries_h, self.long_only, self.bias_side,
            self.signals_csv, self.signals_recent_sec,
            self.min_distance_pct, self.max_distance_pct, self.allow_events,
            self.cooldown_sec, self.max_opens_per_tick, self.max_open_positions,
            self.fee, self.quick_expiry_secs
        )

    # ---------- IO helpers ----------
    def _read_open(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.fp_open)
            # 컬럼 보정
            for c in self.open_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[self.open_cols]
        except Exception:
            return pd.DataFrame(columns=self.open_cols)

    def _write_open(self, df: pd.DataFrame):
        # 컬럼 순서 강제
        df2 = df.copy()
        for c in self.open_cols:
            if c not in df2.columns:
                df2[c] = pd.NA
        df2[self.open_cols].to_csv(self.fp_open, index=False)

    def _append_closed(self, rows: List[ClosedTrade]):
        if not rows:
            return
        df = pd.DataFrame([asdict(r) for r in rows])
        # 컬럼 보정
        for c in self.trades_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[self.trades_cols]
        header_needed = not os.path.exists(self.fp_trades) or os.path.getsize(self.fp_trades) == 0
        df.to_csv(self.fp_trades, mode="a", header=header_needed, index=False)

    # ---------- signals ----------
    def _load_signals(self) -> pd.DataFrame:
        if not self.signals_csv or not os.path.exists(self.signals_csv):
            return pd.DataFrame(columns=["ts","event","side","symbol","distance_pct"])
        try:
            df = pd.read_csv(self.signals_csv)
        except Exception:
            return pd.DataFrame(columns=["ts","event","side","symbol","distance_pct"])

        # ts 파싱
        def parse_ts(x):
            try:
                # csv ts는 ISO-8601(오프셋 포함) 가정
                return datetime.fromisoformat(str(x))
            except Exception:
                return pd.NaT

        df["ts_parsed"] = df["ts"].apply(parse_ts)
        df = df.dropna(subset=["ts_parsed"])
        df = df.sort_values("ts_parsed")
        return df

    def _filter_signals(self, sig_df: pd.DataFrame) -> pd.DataFrame:
        if sig_df.empty:
            return sig_df

        df = sig_df.copy()

        # 이벤트 화이트리스트
        df = df[df["event"].isin(self.allow_events)]

        # long-only 편향
        if self.long_only:
            df = df[df["side"] == self.bias_side]

        # 최근성
        if self.signals_recent_sec > 0:
            cutoff = now_utc() - timedelta(seconds=self.signals_recent_sec)
            df = df[df["ts_parsed"] >= cutoff]

        # 거리 필터
        if (self.min_distance_pct is not None) or (self.max_distance_pct is not None):
            df["distance_pct_num"] = pd.to_numeric(df.get("distance_pct"), errors="coerce")
            df = df[df["distance_pct_num"].notna()]
            if self.min_distance_pct is not None:
                df = df[df["distance_pct_num"] >= float(self.min_distance_pct)]
            if self.max_distance_pct is not None:
                df = df[df["distance_pct_num"] <= float(self.max_distance_pct)]

        # 심볼/타임프레임 중복 제거(가장 최신만)
        df = df.drop_duplicates(subset=["symbol"], keep="last")
        return df

    def _open_from_signals(self) -> int:
        sig_df = self._load_signals()
        cand = self._filter_signals(sig_df)
        if cand.empty:
            return 0

        open_df = self._read_open()
        open_symbols = set(open_df["symbol"].dropna().astype(str)) if not open_df.empty else set()

        opened = 0
        now_s = ts_str(now_utc())
        now_epoch = time.time()

        # 만기(sec) 계산
        def expiry_secs_for(eh: float) -> int:
            return int(self.quick_expiry_secs if self.quick_expiry_secs else eh * 3600)

        # 포지션 수 한도
        current_open = 0 if open_df.empty else len(open_df)
        remaining_capacity = max(0, self.max_open_positions - current_open)
        per_tick_capacity = min(self.max_opens_per_tick, remaining_capacity)
        if per_tick_capacity <= 0:
            return 0

        for _, r in cand.iterrows():
            if opened >= per_tick_capacity:
                break
            sym = str(r.get("symbol", "")).strip()
            if not sym:
                continue

            # 이미 보유 중이면 스킵
            if sym in open_symbols:
                continue

            # 쿨다운
            last_t = self.last_open_ts_by_symbol.get(sym, 0)
            if now_epoch - last_t < self.cooldown_sec:
                continue

            # 진입(만기 리스트마다 1개만) — 가장 짧은 만기 우선
            for eh in sorted(self.expiries_h):
                trade_id = f"S{int(time.time()*1000)}_{sym}_{r.get('event','')}_{str(eh).replace('.','_')}"
                expiry_sec = expiry_secs_for(eh)
                row = OpenTrade(
                    id=trade_id,
                    strategy="tv_signal",
                    expiry_h=eh,
                    side="long",  # long-only
                    entry_ts=now_s,
                    entry_price=100.0,  # 더미 (실계좌 연결 전까지)
                    expiry_sec=expiry_sec,
                    symbol=sym,
                    event=str(r.get("event","")),
                    signal_ts=str(r.get("ts","")),
                    distance_pct=float(r["distance_pct_num"]) if "distance_pct_num" in r and pd.notna(r["distance_pct_num"]) else None,
                )
                new_row = pd.DataFrame([asdict(row)])
                frames = [df for df in (open_df, new_row) if not df.empty]
                open_df = pd.concat(frames, ignore_index=True) if frames else new_row
                opened += 1
                open_symbols.add(sym)
                self.last_open_ts_by_symbol[sym] = now_epoch
                break  # 같은 심볼로 여러 만기 중복 진입 방지

        if opened:
            self._write_open(open_df)
        return opened

    # ---------- exits ----------
    def _close_expired(self) -> int:
        open_df = self._read_open()
        if open_df.empty:
            return 0

        now_dt = now_utc()
        remaining = []
        closed_rows: List[ClosedTrade] = []

        for _, r in open_df.iterrows():
            try:
                entry_dt = datetime.strptime(str(r["entry_ts"]), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                # entry_ts가 비정상인 행은 보존하지 않음
                continue

            expiry_td = timedelta(seconds=int(r["expiry_sec"]))
            if now_dt >= entry_dt + expiry_td:
                # 더미 PnL: ±0.1~0.3%에 수수료 왕복 차감
                drift = random.uniform(0.001, 0.003)
                sign = 1 if random.random() < 0.55 else -1
                gross = sign * drift
                net = gross - 2.0 * self.fee  # 진입/청산 왕복 수수료 가정
                exit_price = float(r["entry_price"]) * (1.0 + net)

                closed = ClosedTrade(
                    id=r["id"],
                    strategy=str(r["strategy"]),
                    expiry_h=float(r["expiry_h"]),
                    side=str(r["side"]),
                    entry_ts=str(r["entry_ts"]),
                    exit_ts=ts_str(now_dt),
                    entry_price=float(r["entry_price"]),
                    exit_price=exit_price,
                    pnl=net,
                    status="expired",
                    symbol=str(r.get("symbol","")),
                    event=str(r.get("event","")),
                    signal_ts=str(r.get("signal_ts","")),
                    distance_pct=float(r.get("distance_pct")) if pd.notna(r.get("distance_pct")) else None,
                )
                closed_rows.append(closed)
            else:
                remaining.append(r)

        if closed_rows:
            self._append_closed(closed_rows)

        # 오픈북 갱신
        if remaining:
            self._write_open(pd.DataFrame(remaining))
        else:
            self._write_open(pd.DataFrame(columns=self.open_cols))

        return len(closed_rows)

    # ---------- main ticks ----------
    def run_once(self) -> Tuple[int, int]:
        opened = self._open_from_signals()
        closed = self._close_expired()
        return opened, closed

    def loop(self, interval_sec: int, stop_at: Optional[datetime] = None):
        logging.info("engine loop started")
        try:
            while self.running:
                opened, closed = self.run_once()
                logging.info("tick summary: opened=%s closed=%s", opened, closed)
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
    # signals & filters
    p.add_argument("--signals-csv", type=str, default=None)
    p.add_argument("--long-only", action="store_true", help="Only open long positions")
    p.add_argument("--bias-side", choices=["resistance", "support"], default="resistance",
                   help="Which side is considered bullish when --long-only (default: resistance)")
    p.add_argument("--signals-recent-sec", type=int, default=180)
    p.add_argument("--min-distance-pct", type=float, default=None)
    p.add_argument("--max-distance-pct", type=float, default=None)
    p.add_argument("--allow-events", type=str, default="box_breakout,line_breakout,price_in_box")
    p.add_argument("--cooldown-sec", type=int, default=300)
    p.add_argument("--max-opens-per-tick", type=int, default=3)
    p.add_argument("--max-open-positions", type=int, default=30)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--reset-open", action="store_true", help="Archive & reset open book at start")
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

    # 이벤트 리스트 파싱
    allow_events = [e.strip() for e in str(args.allow_events).split(",") if e.strip()]

    logging.info(f"expiries_h={expiries_h} {note}")
    if args.long_only:
        logging.info("long_only=True")

    eng = PaperEngine(
        root=root,
        expiries_h=expiries_h,
        quick_expiry_secs=args.quick_expiry_secs,
        interval=args.interval,
        signals_csv=args.signals_csv,
        long_only=args.long_only,
        bias_side=args.bias_side,
        signals_recent_sec=args.signals_recent_sec,
        min_distance_pct=args.min_distance_pct,
        max_distance_pct=args.max_distance_pct,
        allow_events=allow_events,
        cooldown_sec=args.cooldown_sec,
        max_opens_per_tick=args.max_opens_per_tick,
        max_open_positions=args.max_open_positions,
        fee=args.fee,
        reset_open=args.reset_open,
    )

    if args.once:
        opened, closed = eng.run_once()
        logging.info("single step done; opened=%s closed=%s; exiting", opened, closed)
        eng.close()
    elif args.run_for:
        stop_at = now_utc() + timedelta(minutes=args.run_for)
        logging.info(f"run-for: {args.run_for} min (stop_at={stop_at.isoformat()})")
        eng.loop(interval_sec=args.interval, stop_at=stop_at)
    else:
        eng.loop(interval_sec=args.interval)

    logging.info("=== Paper Engine END ===")

if __name__ == "__main__":
    main()