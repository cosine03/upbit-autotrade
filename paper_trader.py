# paper_trader.py
import os, sys, time, csv, json, hashlib, logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import pandas as pd
from datetime import timedelta

# ========= CONFIG =========
SIGNAL_CSV_PATH = r"logs/signals_tv.csv"   # 시그널 CSV (헤더 포함)
SIGNAL_POLL_SEC = 5                        # 시그널 폴링 주기(초)
FEE_BPS = 10                               # 0.10% (왕복은 2번 발생)
SLIPPAGE_BPS = 2                           # 0.02% (진입/청산 시 적용)
DEFAULT_RISK_PCT = 0.01                    # 포지션당 1% 리스크
EXPIRIES_H_DEFAULT = [0.5, 1.0, 2.0]       # 만기 후보 (시간)
PRICE_FEED_CSV = r"logs/price_feed.csv"    # price feed 모드에서 읽을 파일(ts,symbol,price)

STATE_DIR = "state"
LOG_DIR = "logs/paper_trading"
LAST_OFF_FP = os.path.join(STATE_DIR, "last_signal_offset.json")
SEEN_KEYS_FP = os.path.join(STATE_DIR, "seen_signal_keys.json")

TRADES_FP = os.path.join(LOG_DIR, "trades.csv")            # 체결 로그(개별 fill)
POSITIONS_FP = os.path.join(LOG_DIR, "positions.csv")      # 포지션 단위 오픈/클로즈 기록
EQUITY_FP = os.path.join(LOG_DIR, "equity.csv")            # 에쿼티 타임라인
EXPIRY_STATS_FP = os.path.join(LOG_DIR, "expiry_stats.csv")# 만기별 누적 성과

# ========= UTILS =========
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def now_utc():
    # tz-aware UTC timestamp
    return pd.Timestamp.now(tz="UTC")

def bps(v: float) -> float:
    return v / 1e4

def parse_expiries_arg(s: Optional[str]) -> List[float]:
    if not s:
        return EXPIRIES_H_DEFAULT
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower().replace("h","")
        if tok:
            out.append(float(tok))
    return out or EXPIRIES_H_DEFAULT

def to_float(x, default=0.0):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def sig_key(row: Dict) -> str:
    base = f"{row.get('symbol','')}|{row.get('event','')}|{row.get('ts','')}|{row.get('id','')}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

# ========= SIGNAL FEED =========
class SignalFeed:
    def __init__(self, csv_path=SIGNAL_CSV_PATH, state_dir=STATE_DIR, last_off_fp=LAST_OFF_FP, seen_fp=SEEN_KEYS_FP):
        self.csv_path = csv_path
        self.last_off_fp = last_off_fp
        self.seen_fp = seen_fp
        ensure_dirs(state_dir)
        self._offset = 0
        self._seen = set()
        self._load_state()

    def _load_state(self):
        try:
            with open(self.last_off_fp, "r", encoding="utf-8") as f:
                self._offset = json.load(f).get("offset", 0)
        except Exception:
            self._offset = 0
        try:
            with open(self.seen_fp, "r", encoding="utf-8") as f:
                self._seen = set(json.load(f).get("seen", []))
        except Exception:
            self._seen = set()

    def _save_state(self):
        with open(self.last_off_fp, "w", encoding="utf-8") as f:
            json.dump({"offset": self._offset}, f)
        with open(self.seen_fp, "w", encoding="utf-8") as f:
            json.dump({"seen": list(self._seen)}, f)

    def poll(self) -> List[Dict]:
        """파일 뒤에서부터 신규 레코드 읽어 반환"""
        if not os.path.exists(self.csv_path):
            return []
        new = []
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            f.seek(self._offset, os.SEEK_SET)
            reader = csv.DictReader(f)
            for row in reader:
                k = sig_key(row)
                if k in self._seen:
                    continue
                self._seen.add(k)
                new.append(row)
            self._offset = f.tell()
        if new:
            self._save_state()
        return new

# ========= PRICE SOURCE =========
class PriceSource:
    """
    price_mode:
      - const    : 1.0 고정 (기본)
      - signals  : 시그널 row의 price 컬럼 사용(없으면 1.0)
      - feed     : logs/price_feed.csv 최신가 사용(없으면 직전가 또는 1.0)
    """
    def __init__(self, mode="const", feed_csv=PRICE_FEED_CSV):
        self.mode = mode
        self.feed_csv = feed_csv
        self.last_price_by_sym: Dict[str, float] = {}

    def from_signal(self, sym: str, sig_row: Dict) -> float:
        if self.mode == "signals":
            px = to_float(sig_row.get("price"), default=1.0)
            if px > 0:
                self.last_price_by_sym[sym] = px
                return px
        if self.mode == "feed":
            px = self._from_feed(sym)
            if px > 0:
                self.last_price_by_sym[sym] = px
                return px
        # const or fallback
        px = self.last_price_by_sym.get(sym, 1.0 if self.mode=="const" else 1.0)
        self.last_price_by_sym[sym] = px
        return px

    def latest(self, sym: str) -> float:
        if self.mode == "feed":
            px = self._from_feed(sym)
            if px > 0:
                self.last_price_by_sym[sym] = px
                return px
        return self.last_price_by_sym.get(sym, 1.0 if self.mode=="const" else 1.0)

    def _from_feed(self, sym: str) -> float:
        try:
            if not os.path.exists(self.feed_csv):
                return self.last_price_by_sym.get(sym, 1.0)
            df = pd.read_csv(self.feed_csv)
            df = df[df["symbol"] == sym]
            if df.empty:
                return self.last_price_by_sym.get(sym, 1.0)
            px = float(df.iloc[-1]["price"])
            return px if px > 0 else self.last_price_by_sym.get(sym, 1.0)
        except Exception:
            return self.last_price_by_sym.get(sym, 1.0)

# ========= DATA MODELS =========
@dataclass
class Position:
    id: str
    symbol: str
    side: str              # BUY / SELL
    qty: float
    entry_px: float
    entry_ts: pd.Timestamp
    expiry_h: float
    expiry_ts: pd.Timestamp
    fee_entry: float
    meta: Dict

@dataclass
class CloseResult:
    pos_id: str
    symbol: str
    side: str
    qty: float
    entry_px: float
    exit_px: float
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    expiry_h: float
    pnl: float
    fee_entry: float
    fee_exit: float
    meta: Dict

# ========= PAPER ENGINE =========
class PaperEngine:
    def __init__(self, root_dir=".", shadow=False, interval_sec=10,
                 price_mode="const", expiries: Optional[List[float]]=None):
        ensure_dirs(LOG_DIR, STATE_DIR)
        self.root = root_dir
        self.shadow = shadow
        self.interval_sec = interval_sec
        self.log = logging.getLogger("PaperEngine")
        self.log.setLevel(logging.INFO)
        if not self.log.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.log.addHandler(ch)

        self._equity = 1.0
        self.signal_feed = SignalFeed()
        self.price = PriceSource(mode=price_mode)
        self.expiries = expiries or EXPIRIES_H_DEFAULT
        self.open_positions: Dict[str, Position] = {}

        # 파일 초기화
        if not os.path.exists(EQUITY_FP):
            pd.DataFrame([{"ts": now_utc(), "equity": self._equity}]).to_csv(EQUITY_FP, index=False)
        if not os.path.exists(EXPIRY_STATS_FP):
            pd.DataFrame([{
                "date": pd.Timestamp.now(tz="UTC").date(),
                "expiry_h": "init",
                "trades": 0, "wins": 0, "win_rate": 0.0, "avg_net": 0.0, "total_net": 0.0
            }]).to_csv(EXPIRY_STATS_FP, index=False)

    # ---- portfolio sizing ----
    def _size_from_equity(self, price):
        eq = float(self._equity)
        qty = max(eq * DEFAULT_RISK_PCT / max(price, 1e-9), 0.0)
        return round(qty, 6)

    # ---- open/close ----
    def _open_position(self, sym, side, qty, px, ts, expiry_h, meta):
        fee = px * qty * bps(FEE_BPS)
        pos_id = hashlib.md5(f"{sym}|{ts}|{px}|{expiry_h}|{side}".encode("utf-8")).hexdigest()[:16]
        pos = Position(
            id=pos_id, symbol=sym, side=side, qty=qty, entry_px=px, entry_ts=ts,
            expiry_h=expiry_h, expiry_ts=ts + timedelta(hours=expiry_h),
            fee_entry=fee, meta=meta or {}
        )
        self.open_positions[pos_id] = pos
        # trade log (entry)
        entry_row = {
            "ts": ts, "pos_id": pos_id, "symbol": sym, "side": side, "qty": qty,
            "price": px, "fee": fee, "type": "ENTRY", "expiry_h": expiry_h
        }
        pd.DataFrame([entry_row]).to_csv(TRADES_FP, mode="a", header=not os.path.exists(TRADES_FP), index=False)
        if not self.shadow:
            self._equity -= fee
            pd.DataFrame([{"ts": ts, "equity": self._equity}]).to_csv(EQUITY_FP, mode="a", header=False, index=False)
        self.log.info(f"OPEN {side} {sym} qty={qty} px={px:.6f} fee={fee:.6f} exp={expiry_h}h")

    def _close_position(self, pos: Position, exit_px: float, exit_ts: pd.Timestamp) -> CloseResult:
        fee_exit = exit_px * pos.qty * bps(FEE_BPS)
        sign = 1 if pos.side == "BUY" else -1
        pnl_gross = (exit_px - pos.entry_px) * pos.qty * sign
        pnl_net = pnl_gross - pos.fee_entry - fee_exit - (pos.entry_px * pos.qty * bps(SLIPPAGE_BPS)) - (exit_px * pos.qty * bps(SLIPPAGE_BPS))
        cr = CloseResult(
            pos_id=pos.id, symbol=pos.symbol, side=pos.side, qty=pos.qty,
            entry_px=pos.entry_px, exit_px=exit_px, entry_ts=pos.entry_ts, exit_ts=exit_ts,
            expiry_h=pos.expiry_h, pnl=pnl_net, fee_entry=pos.fee_entry, fee_exit=fee_exit, meta=pos.meta
        )
        # trade log (exit)
        exit_row = {
            "ts": exit_ts, "pos_id": pos.id, "symbol": pos.symbol, "side": pos.side, "qty": pos.qty,
            "price": exit_px, "fee": fee_exit, "type": "EXIT", "expiry_h": pos.expiry_h
        }
        pd.DataFrame([exit_row]).to_csv(TRADES_FP, mode="a", header=not os.path.exists(TRADES_FP), index=False)
        # position log
        pd.DataFrame([{
            "pos_id": cr.pos_id, "symbol": cr.symbol, "side": cr.side, "qty": cr.qty,
            "entry_px": cr.entry_px, "exit_px": cr.exit_px,
            "entry_ts": cr.entry_ts, "exit_ts": cr.exit_ts,
            "expiry_h": cr.expiry_h, "pnl": cr.pnl
        }]).to_csv(POSITIONS_FP, mode="a", header=not os.path.exists(POSITIONS_FP), index=False)
        if not self.shadow:
            self._equity += cr.pnl - 0.0  # pnl에는 수수료/슬리피지 반영됨
            pd.DataFrame([{"ts": exit_ts, "equity": self._equity}]).to_csv(EQUITY_FP, mode="a", header=False, index=False)
        self.log.info(f"CLOSE {pos.side} {pos.symbol} exp={pos.expiry_h}h exit_px={exit_px:.6f} pnl={cr.pnl:.6f}")
        return cr

    # ---- stats ----
    def _update_expiry_stats(self, closes: List[CloseResult]):
        if not closes:
            return
        # 날짜 단위 집계 (UTC)
        df = pd.DataFrame([{
            "date": pd.Timestamp(c.exit_ts).tz_convert("UTC").date(),
            "expiry_h": c.expiry_h,
            "pnl": c.pnl
        } for c in closes])
        grp = df.groupby(["date", "expiry_h"]).agg(
            trades=("pnl","count"),
            wins=("pnl", lambda s: (s > 0).sum()),
            avg_net=("pnl","mean"),
            total_net=("pnl","sum")
        ).reset_index()
        # 기존 파일 로드 → 누적 업데이트
        if os.path.exists(EXPIRY_STATS_FP):
            cur = pd.read_csv(EXPIRY_STATS_FP)
        else:
            cur = pd.DataFrame(columns=["date","expiry_h","trades","wins","win_rate","avg_net","total_net"])
        for _, r in grp.iterrows():
            mask = (cur.get("date", pd.Series([])) == str(r["date"])) & (cur.get("expiry_h", pd.Series([])) == r["expiry_h"])
            if mask.any():
                idx = cur[mask].index[0]
                cur.loc[idx, "trades"] = cur.loc[idx, "trades"] + int(r["trades"])
                cur.loc[idx, "wins"]   = cur.loc[idx, "wins"] + int(r["wins"])
                cur.loc[idx, "avg_net"] = ( (cur.loc[idx, "avg_net"] * (cur.loc[idx, "trades"]-int(r["trades"]))) + (r["avg_net"] * r["trades"]) ) / max(cur.loc[idx, "trades"],1)
                cur.loc[idx, "total_net"] = cur.loc[idx, "total_net"] + r["total_net"]
            else:
                cur = pd.concat([cur, pd.DataFrame([{
                    "date": str(r["date"]),
                    "expiry_h": r["expiry_h"],
                    "trades": int(r["trades"]),
                    "wins": int(r["wins"]),
                    "avg_net": float(r["avg_net"]),
                    "total_net": float(r["total_net"])
                }])], ignore_index=True)
        # win_rate 업데이트
        cur["win_rate"] = cur.apply(lambda x: (x["wins"]/x["trades"]) if x["trades"] else 0.0, axis=1)
        cur.to_csv(EXPIRY_STATS_FP, index=False)

    # ---- mapping ----
    def _map_signal_to_orders(self, sig: Dict):
        sym = sig.get("symbol", "KRW-BTC")
        event = (sig.get("event","") or "").lower()
        # 기본 방향: breakout=BUY, breakdown=SELL, 기타는 BUY
        side = "BUY" if "breakout" in event else ("SELL" if "breakdown" in event else "BUY")
        # 진입가
        entry_px = self.price.from_signal(sym, sig)
        qty = self._size_from_equity(entry_px)
        ts = now_utc()
        for exp_h in self.expiries:
            self._open_position(sym, side, qty, entry_px, ts, exp_h, meta=sig)

    # ---- loop ----
    def step(self):
        # 1) 새 시그널 → 포지션 오픈(만기별로 다중)
        sigs = self.signal_feed.poll()
        if sigs:
            self.log.info(f"signals: {len(sigs)} (fan-out x{len(self.expiries)})")
            for s in sigs:
                try:
                    self._map_signal_to_orders(s)
                except Exception as e:
                    self.log.error(f"signal->orders error: {e}")

        # 2) 만기 도달 포지션 청산
        if self.open_positions:
            now = now_utc()
            to_close = []
            for pid, pos in list(self.open_positions.items()):
                if now >= pos.expiry_ts:
                    to_close.append(pid)
            closes: List[CloseResult] = []
            for pid in to_close:
                pos = self.open_positions.pop(pid, None)
                if not pos: 
                    continue
                exit_px = self.price.latest(pos.symbol)
                cr = self._close_position(pos, exit_px, now)
                closes.append(cr)
            # 만기별 성과 누적
            self._update_expiry_stats(closes)

    def run(self, run_for_min=None):
        self.log.info("=== Paper Engine START ===")
        start = now_utc()
        stop_at = None
        if run_for_min:
            stop_at = start + timedelta(minutes=run_for_min)
            self.log.info(f"run-for: {run_for_min} min (stop_at={stop_at})")
        try:
            while True:
                self.step()
                time.sleep(self.interval_sec)
                if stop_at and now_utc() >= stop_at:
                    self.log.info("timebox reached; exiting")
                    break
        except KeyboardInterrupt:
            self.log.info("KeyboardInterrupt - shutting down...")
        self.log.info("engine closed")
        self.log.info("=== Paper Engine END ===")

# ========= CLI =========
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="한 번만 step 실행")
    p.add_argument("--run-for", type=int, help="분 단위로 실행 유지")
    p.add_argument("--interval", type=int, default=10, help="폴링 주기(초)")
    p.add_argument("--shadow", action="store_true", help="equity 갱신 없이 로직만")
    p.add_argument("--price-mode", type=str, default="const", choices=["const","signals","feed"], help="가격 소스")
    p.add_argument("--expiries", type=str, help='예: "0.5,1,2"')
    args = p.parse_args()

    expiries = parse_expiries_arg(args.expiries)
    eng = PaperEngine(interval_sec=args.interval, shadow=args.shadow,
                      price_mode=args.price_mode, expiries=expiries)
    if args.once:
        eng.step()
    else:
        eng.run(run_for_min=args.run_for)

if __name__ == "__main__":
    main()