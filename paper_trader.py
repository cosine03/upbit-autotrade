# -*- coding: utf-8 -*-
import argparse
import csv
import logging
import os
import sys
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

# ========= 경로/상수 =========
ROOT = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(ROOT, "logs", "paper")
SIGNALS_CSV_DEFAULT = os.path.join(ROOT, "logs", "signals_tv.csv")
UNIVERSE_TXT_DEFAULT = os.path.join(ROOT, "configs", "universe.txt")

ENGINE_LOG_PATH = os.path.join(LOGS_DIR, "engine.log")
ENGINE_START_TXT = os.path.join(LOGS_DIR, "engine_start.txt")

TRADES_OPEN_PATH = os.path.join(LOGS_DIR, "trades_open.csv")
TRADES_CLOSED_PATH = os.path.join(LOGS_DIR, "trades_closed.csv")
REJECTS_PATH = os.path.join(LOGS_DIR, "rejects.csv")

OPEN_HEADER = ["opened_at","symbol","event","side","level","expire_at","fee"]
CLOSED_HEADER = ["opened_at","symbol","event","side","level",
                 "closed_at","entry_price","exit_price","pnl","reason","fee"]
REJECTS_HEADER = ["ts","symbol","event","side","level","distance_pct",
                  "phase","reason","engine_start","now"]

ALLOWED_EVENTS_DEFAULT = {"line_breakout","box_breakout"}  # PIB는 기본 reject

# ========= 유틸 =========
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def ensure_csv_header(path: str, header: List[str]):
    ensure_dir(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        return
    # 첫 줄이 헤더인지 간단 확인
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    want = ",".join(header)
    if first != want:
        # 기존 파일이 꼬여있으면 헤더 재기록(데이터 보존은 여기선 스킵)
        # 필요시 별도 fix 스크립트로 처리
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            rdr = list(csv.reader(f))
            if rdr:
                # 첫 줄이 데이터처럼 보이면 전부 보존
                if rdr and rdr[0] and len(rdr[0]) > 0 and rdr[0][0].startswith("202"):
                    rows = rdr
                elif len(rdr) > 1:
                    rows = rdr[1:]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

def write_engine_start():
    ensure_dir(ENGINE_START_TXT)
    with open(ENGINE_START_TXT, "w", encoding="utf-8") as f:
        f.write(iso(utcnow()).strip() + "\n")

def setup_logging():
    ensure_dir(ENGINE_LOG_PATH)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ENGINE_LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def load_universe(path: str) -> Optional[Set[str]]:
    if not os.path.exists(path):
        return None
    syms = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            syms.add(s)
    return syms if syms else None

# ========= 데이터 모델 =========
@dataclass
class Signal:
    ts: datetime
    symbol: str
    event: str
    side: str
    level: int
    distance_pct: float

@dataclass
class Position:
    opened_at: datetime
    symbol: str
    event: str
    side: str
    level: int
    expire_at: datetime
    entry_price: Optional[float] = None  # price_feeder 붙이면 채움
    # TP/SL 자리
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None

# ========= CSV 쓰기 =========
def write_open(p: Position, fee: float):
    ensure_csv_header(TRADES_OPEN_PATH, OPEN_HEADER)
    with open(TRADES_OPEN_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            iso(p.opened_at), p.symbol, p.event, p.side, p.level, iso(p.expire_at), fee
        ])

def write_closed(p: Position, closed_at: datetime, reason: str,
                 exit_price: Optional[float], fee: float):
    ensure_csv_header(TRADES_CLOSED_PATH, CLOSED_HEADER)
    # pnl 계산은 price_feeder 도입 후
    pnl = None
    entry_price = p.entry_price
    with open(TRADES_CLOSED_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            iso(p.opened_at), p.symbol, p.event, p.side, p.level,
            iso(closed_at),
            "" if entry_price is None else entry_price,
            "" if exit_price is None else exit_price,
            "" if pnl is None else pnl,
            reason,
            fee
        ])

def write_reject(sig: Signal, phase: str, reason: str, engine_start: datetime):
    ensure_csv_header(REJECTS_PATH, REJECTS_HEADER)
    with open(REJECTS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            iso(sig.ts), sig.symbol, sig.event, sig.side, sig.level,
            "" if sig.distance_pct is None else sig.distance_pct,
            phase, reason, iso(engine_start), iso(utcnow())
        ])

# ========= 시그널 로딩 =========
def parse_float_safe(x: str, default: float) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def parse_int_safe(x: str, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def load_recent_signals(signals_csv: str, recent: timedelta) -> List[Signal]:
    res: List[Signal] = []
    if not os.path.exists(signals_csv):
        return res
    threshold = utcnow() - recent
    with open(signals_csv, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    # signals_tv.csv 포맷(헤더 없음 가정):
    # ts,event,side,level,touches,symbol,timeframe,extra,host,message
    # 또는 (우리가 저장한 버전) 동일 컬럼 순서
    for i, row in enumerate(rows):
        if i == 0 and row and not row[0].startswith("202"):
            # 헤더로 보이면 스킵
            continue
        if not row or len(row) < 6:
            continue
        try:
            ts_s = row[0]
            event = row[1]
            side = row[2]
            level = parse_int_safe(row[3], 0)
            symbol = row[5]
            # distance_pct 컬럼이 없으므로 규칙에 따라 보정:
            #  - breakout류: 0.0 간주
            #  - price_in_box도 0.0 기록(미래 확장 대비)
            dist = 0.0
            ts = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
            if ts >= threshold:
                res.append(Signal(ts=ts, symbol=symbol, event=event,
                                  side=side, level=level, distance_pct=dist))
        except Exception:
            continue
    return res

# ========= 엔진 =========
class PaperEngine:
    def __init__(
        self,
        signals_csv: str,
        universe_txt: Optional[str],
        allow_events: Set[str],
        long_only: bool,
        min_distance: float,
        max_distance: float,
        cooldown_sec: int,
        max_opens_per_tick: int,
        max_open_positions: int,
        expiry_min: int,
        interval_sec: int,
        fee: float
    ):
        self.signals_csv = signals_csv
        self.universe = load_universe(universe_txt) if universe_txt else None
        self.allow_events = allow_events
        self.long_only = long_only
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.cooldown = timedelta(seconds=cooldown_sec)
        self.max_opens_per_tick = max_opens_per_tick
        self.max_open_positions = max_open_positions
        self.expiry = timedelta(minutes=expiry_min)
        self.interval = interval_sec
        self.fee = fee

        self.engine_start = utcnow()
        write_engine_start()

        ensure_csv_header(TRADES_OPEN_PATH, OPEN_HEADER)
        ensure_csv_header(TRADES_CLOSED_PATH, CLOSED_HEADER)
        ensure_csv_header(REJECTS_PATH, REJECTS_HEADER)

        self.positions: Dict[str, Position] = {}  # symbol -> Position (심플: 심볼당 1개)
        self.last_open_time: Dict[str, datetime] = {}  # cooldown
        self.seen_keys: Set[Tuple[str,str,str,str,int]] = set()  # (ts,symbol,event,side,level)

        logging.info("expiries=['fixed'] (%dm)", expiry_min)
        logging.info("long_only=%s", long_only)
        logging.info(
            "PaperEngine initialized allow_events=%s, recent=60s, dist=(%g,%g), cooldown=%ss, max_per_tick=%d, max_open_positions=%d, fee=%g",
            sorted(list(allow_events)),
            min_distance, max_distance,
            int(self.cooldown.total_seconds()),
            max_opens_per_tick,
            max_open_positions,
            fee
        )

    def _symbol_allowed(self, symbol: str) -> bool:
        if self.universe is None:
            return True
        return symbol in self.universe

    def _passes_filters(self, sig: Signal) -> Optional[str]:
        # event
        if sig.event not in self.allow_events:
            return "event_not_allowed"
        # long_only: resistance만
        if self.long_only and sig.side != "resistance":
            return "side_not_allowed"
        # distance
        d = sig.distance_pct
        if d is None:
            return "distance_missing"
        if d < self.min_distance or d > self.max_distance:
            return "distance_out_of_range"
        # universe
        if not self._symbol_allowed(sig.symbol):
            return "symbol_not_in_universe"
        # cooldown
        last = self.last_open_time.get(sig.symbol)
        if last and (utcnow() - last) < self.cooldown:
            return "cooldown"
        # capacity
        if len(self.positions) >= self.max_open_positions:
            return "capacity_reached"
        return None

    def _try_open(self, sigs: List[Signal]) -> int:
        opened = 0
        for sig in sigs:
            if opened >= self.max_opens_per_tick:
                break
            key = (iso(sig.ts), sig.symbol, sig.event, sig.side, sig.level)
            if key in self.seen_keys:
                continue
            self.seen_keys.add(key)

            reason = self._passes_filters(sig)
            if reason is not None:
                write_reject(sig, phase="run", reason=reason, engine_start=self.engine_start)
                continue

            # 오픈
            opened_at = utcnow()
            expire_at = opened_at + self.expiry
            pos = Position(
                opened_at=opened_at,
                symbol=sig.symbol,
                event=sig.event,
                side=sig.side,
                level=sig.level,
                expire_at=expire_at,
                entry_price=None,  # price_feeder 붙이면 채움
                take_profit_pct=None,
                stop_loss_pct=None,
            )
            self.positions[sig.symbol] = pos
            self.last_open_time[sig.symbol] = opened_at
            write_open(pos, fee=self.fee)
            opened += 1
        return opened

    def _try_close_time_expired(self) -> int:
        now = utcnow()
        closed = 0
        to_close = [sym for sym, p in self.positions.items() if now >= p.expire_at]
        for sym in to_close:
            p = self.positions.pop(sym)
            write_closed(p, closed_at=now, reason="time_expired", exit_price=None, fee=self.fee)
            closed += 1
        return closed

    def tick(self) -> Tuple[int,int,Dict[str,int]]:
        # 최근 60초 신호만 본다(중복 방지를 위해 seen_keys 사용)
        sigs = load_recent_signals(self.signals_csv, recent=timedelta(seconds=60))

        # 사전 집계(왜 reject 나갔는지 확인하기 좋게)
        rej_counts = {
            "pre": 0, "stale": 0,
            "event": 0, "side": 0, "dist": 0, "cool": 0, "cap": 0, "univ": 0, "miss": 0
        }

        # 필터 통과/리젝트 처리 및 오픈
        considered = []
        for s in sigs:
            # 중복 신호(동일 키)면 skip(여기선 stale 로는 잡지 않음)
            k = (iso(s.ts), s.symbol, s.event, s.side, s.level)
            if k in self.seen_keys:
                # stale 카운트만 올리고 끝
                rej_counts["stale"] += 1
                continue
            considered.append(s)

        opened = 0
        opens_this_tick = []
        for s in considered:
            reason = self._passes_filters(s)
            if reason:
                write_reject(s, "run", reason, self.engine_start)
                if reason == "event_not_allowed": rej_counts["event"] += 1
                elif reason == "side_not_allowed": rej_counts["side"] += 1
                elif reason == "distance_out_of_range": rej_counts["dist"] += 1
                elif reason == "cooldown": rej_counts["cool"] += 1
                elif reason == "capacity_reached": rej_counts["cap"] += 1
                elif reason == "symbol_not_in_universe": rej_counts["univ"] += 1
                elif reason == "distance_missing": rej_counts["miss"] += 1
                self.seen_keys.add((iso(s.ts), s.symbol, s.event, s.side, s.level))
                continue

            opens_this_tick.append(s)

        if opens_this_tick:
            opened = self._try_open(opens_this_tick)

        closed = self._try_close_time_expired()

        # 콘솔 요약
        logging.info(
            "tick summary: opened=%d closed=%d rejects_pre=%d rejects_stale=%d rej_event=%d rej_side=%d rej_dist=%d rej_cool=%d rej_cap=%d rej_univ=%d rej_miss=%d",
            opened, closed, 0, rej_counts["stale"], rej_counts["event"],
            rej_counts["side"], rej_counts["dist"], rej_counts["cool"], rej_counts["cap"],
            rej_counts["univ"], rej_counts["miss"]
        )
        return opened, closed, rej_counts

    def run(self, run_for_min: Optional[int]):
        logging.info("=== Paper Engine START === root=%s", ROOT)
        start = utcnow()
        try:
            if run_for_min and run_for_min > 0:
                stop_at = start + timedelta(minutes=run_for_min)
                logging.info("run-for: %d min (stop_at=%s)", run_for_min, iso(stop_at))
            else:
                stop_at = None
                logging.info("run-for: unlimited")

            while True:
                self.tick()
                if stop_at and utcnow() >= stop_at:
                    logging.info("timebox reached; exiting")
                    break
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt - shutting down...")
        finally:
            logging.info("engine closed")
            logging.info("=== Paper Engine END ===")

# ========= CLI =========
def parse_args():
    p = argparse.ArgumentParser(description="Paper Trader Engine")
    p.add_argument("--signals-csv", default=SIGNALS_CSV_DEFAULT)
    p.add_argument("--universe", default=UNIVERSE_TXT_DEFAULT)
    p.add_argument("--allow-events", default="line_breakout,box_breakout")
    p.add_argument("--long-only", action="store_true", default=True)
    p.add_argument("--min-distance-pct", type=float, default=0.0)
    p.add_argument("--max-distance-pct", type=float, default=0.00025)
    p.add_argument("--cooldown-sec", type=int, default=300)
    p.add_argument("--max-opens-per-tick", type=int, default=3)
    p.add_argument("--max-open-positions", type=int, default=30)
    p.add_argument("--expiry-min", type=int, default=30)  # 백테스트 기준
    p.add_argument("--interval-sec", type=int, default=10)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--run-for-min", type=int, default=0,
                   help="0이면 무제한")
    return p.parse_args()

def main():
    setup_logging()
    args = parse_args()

    allow_events = {s.strip() for s in args.allow_events.split(",") if s.strip()}

    eng = PaperEngine(
        signals_csv=args.signals_csv,
        universe_txt=args.universe,
        allow_events=allow_events,
        long_only=args.long_only,
        min_distance=args.min_distance_pct,
        max_distance=args.max_distance_pct,
        cooldown_sec=args.cooldown_sec,
        max_opens_per_tick=args.max_opens_per_tick,
        max_open_positions=args.max_open_positions,
        expiry_min=args.expiry_min,
        interval_sec=args.interval_sec,
        fee=args.fee
    )
    eng.run(run_for_min=(args.run_for_min if args.run_for_min and args.run_for_min > 0 else None))

if __name__ == "__main__":
    main()