# -*- coding: utf-8 -*-
"""
Paper Trading Engine (skeleton)
- 시그널 CSV를 주기적으로 폴링
- 로컬 OHLCV(csv)로 엔트리/TP/SL/만기 판정
- 포지션 오픈/클로즈/성능 집계 CSV 저장

디렉토리(예):
  logs/
    realtime/
      2025-09-28/
        orders_open.csv
        orders_closed.csv
        fills.csv
        equity_curve.csv
        runner.log
"""

import argparse
import time
import os
import sys
import math
import json
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

import pandas as pd
pd.options.mode.chained_assignment = None


# ---------- Utils ----------
def utc_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


# ---------- Data Models ----------
@dataclass
class Signal:
    symbol: str
    ts: pd.Timestamp          # 시그널 발생 시각 (UTC)
    event: str                # 'box_breakout' | 'line_breakout' | ...
    side: str                 # 'resistance' or 'support' (기본: resistance=SHORT, support=LONG)
    touches: Optional[int] = None
    distance_pct: Optional[float] = None
    raw: Dict = None          # 원본 레코드 (필요시 참고)

@dataclass
class Position:
    id: str
    symbol: str
    side: str                 # 'long' | 'short'
    event: str
    ts_sig: pd.Timestamp
    ts_entry: pd.Timestamp
    entry_px: float
    tp_pct: float
    sl_pct: float
    fee_roundtrip: float
    expiry_h: float
    status: str = "open"      # 'open' | 'closed'
    reason: Optional[str] = None  # 'tp' | 'sl' | 'expiry'
    exit_px: Optional[float] = None
    ts_exit: Optional[pd.Timestamp] = None
    gross_ret: Optional[float] = None   # (exit_px/entry_px-1) * (+/-)
    net_ret: Optional[float] = None     # gross - fee_roundtrip

# ---------- Engine ----------
class PaperEngine:
    def __init__(
        self,
        outdir: str,
        signals_csv: str,
        timeframe: str = "15m",
        ohlcv_roots: List[str] = None,
        ohlcv_patterns: List[str] = None,
        entry_policy: str = "prev_close",   # 'prev_close' | 'next_open'
        tp_pct: float = 0.0175,
        sl_pct: float = 0.007,
        fee_roundtrip: float = 0.001,       # 왕복 수수료 (예: 0.1% = 0.001)
        expiries_h: List[float] = None,
        poll_secs: int = 10,
        max_new_per_cycle: int = 50,
        map_resistance_short: bool = True,  # resistance=SHORT, support=LONG 매핑
        dist_max: Optional[float] = None,   # distance_pct 필터 (없으면 스킵)
    ):
        self.outdir = outdir
        ensure_dir(self.outdir)

        self.signals_csv = signals_csv
        self.timeframe = timeframe
        self.ohlcv_roots = ohlcv_roots or [".", "./data", "./data/ohlcv", "./ohlcv", "./logs", "./logs/ohlcv"]
        self.ohlcv_patterns = ohlcv_patterns or [
            "data/ohlcv/{symbol}-{timeframe}.csv",
            "data/ohlcv/{symbol}_{timeframe}.csv",
            "{symbol}-{timeframe}.csv",
            "{symbol}_{timeframe}.csv",
        ]
        self.entry_policy = entry_policy
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.fee_roundtrip = fee_roundtrip
        self.expiries_h = expiries_h or [0.5, 1.0, 2.0]
        self.poll_secs = poll_secs
        self.max_new_per_cycle = max_new_per_cycle
        self.map_resistance_short = map_resistance_short
        self.dist_max = dist_max

        # state
        self._seen_keys = set()  # (symbol, ts_sig, event, side)
        self.open_positions: Dict[str, Position] = {}  # id -> Position

        # outputs
        self.fp_orders_open = os.path.join(self.outdir, "orders_open.csv")
        self.fp_orders_closed = os.path.join(self.outdir, "orders_closed.csv")
        self.fp_fills = os.path.join(self.outdir, "fills.csv")
        self.fp_equity = os.path.join(self.outdir, "equity_curve.csv")
        self.fp_log = os.path.join(self.outdir, "runner.log")

        if not os.path.exists(self.fp_equity):
            pd.DataFrame([{"ts": now_utc(), "equity": 1.0}]).to_csv(self.fp_equity, index=False)

        self._log(f"INIT outdir={self.outdir}")

    # ---------- logging ----------
    def _log(self, msg: str):
        ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.fp_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---------- IO ----------
    def _append_csv(self, path: str, rows: List[dict]):
        df = pd.DataFrame(rows)
        header = not os.path.exists(path)
        df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")

    def _read_signals(self) -> List[Signal]:
        if not os.path.exists(self.signals_csv):
            return []
        d = pd.read_csv(self.signals_csv)
        # 필수 컬럼 heuristic
        req = {"symbol", "ts", "event", "side"}
        if not req.issubset(set(c.lower() for c in d.columns)):
            # 대소문자 편의: 실제 이름 매핑
            rename = {}
            for c in d.columns:
                lc = c.lower()
                if lc in req and lc != c:
                    rename[c] = lc
            if rename:
                d = d.rename(columns=rename)
        # parse
        out: List[Signal] = []
        for _, r in d.iterrows():
            try:
                sym = str(r["symbol"])
                ts_sig = utc_ts(r["ts"])
                event = str(r["event"])
                side = str(r["side"])
                touches = int(r["touches"]) if "touches" in r else None
                dist = float(r["distance_pct"]) if "distance_pct" in r and not pd.isna(r["distance_pct"]) else None
                if self.dist_max is not None and dist is not None and dist > self.dist_max:
                    continue
                if pd.isna(ts_sig):
                    continue
                out.append(Signal(symbol=sym, ts=ts_sig, event=event, side=side,
                                  touches=touches, distance_pct=dist, raw=r.to_dict()))
            except Exception:
                continue
        return out

    def _load_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        for root in self.ohlcv_roots:
            for pat in self.ohlcv_patterns:
                p = os.path.join(root, pat.format(symbol=symbol, timeframe=self.timeframe))
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    if "ts" not in df.columns:
                        return None
                    df["ts"] = utc_ts(df["ts"])
                    # floatify
                    for c in ["open", "high", "low", "close"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
                    return df
        return None

    # ---------- Core logic ----------
    def _entry_from_signal(self, ohlcv: pd.DataFrame, ts_sig: pd.Timestamp) -> Tuple[Optional[int], Optional[pd.Timestamp], Optional[float]]:
        ts_idx = pd.DatetimeIndex(ohlcv["ts"])
        idx = int(ts_idx.searchsorted(ts_sig, side="right") - 1)
        if idx < 0 or idx >= len(ohlcv):
            return None, None, None

        if self.entry_policy == "prev_close":
            ts_entry = ohlcv["ts"].iloc[idx]
            px = float(ohlcv["close"].iloc[idx])
            return idx, ts_entry, px
        elif self.entry_policy == "next_open":
            if idx + 1 >= len(ohlcv):
                return None, None, None
            ts_entry = ohlcv["ts"].iloc[idx + 1]
            px = float(ohlcv["open"].iloc[idx + 1])
            return idx + 1, ts_entry, px
        else:
            return None, None, None

    def _scan_exit(self, ohlcv: pd.DataFrame, i_entry: int, entry_px: float,
                   ts_entry: pd.Timestamp, side: str, expiry_h: float) -> Tuple[pd.Timestamp, float, str]:
        """TP/SL/만기 우선순위: TP/SL 먼저, 없으면 만기 종가."""
        end_ts = ts_entry + pd.Timedelta(hours=expiry_h)
        look = ohlcv.iloc[i_entry + 1:].copy()
        if look.empty:
            return ts_entry, entry_px, "expiry"

        look = look[look["ts"] <= end_ts]
        if look.empty:
            return ts_entry, entry_px, "expiry"

        # 방향
        is_long = (side == "long")

        tp_up = entry_px * (1 + self.tp_pct)
        sl_dn = entry_px * (1 - self.sl_pct)
        tp_dn = entry_px * (1 - self.tp_pct)
        sl_up = entry_px * (1 + self.sl_pct)

        for _, r in look.iterrows():
            hi = float(r["high"])
            lo = float(r["low"])
            tsb = r["ts"]

            if is_long:
                # TP 먼저
                if hi >= tp_up:
                    return tsb, tp_up, "tp"
                if lo <= sl_dn:
                    return tsb, sl_dn, "sl"
            else:
                # short: 가격 하락이 이익
                if lo <= tp_dn:
                    return tsb, tp_dn, "tp"
                if hi >= sl_up:
                    return tsb, sl_up, "sl"

        # 만기: 마지막 바의 종가
        last = look.iloc[-1]
        return last["ts"], float(last["close"]), "expiry"

    def _gross_ret(self, entry_px: float, exit_px: float, side: str) -> float:
        if side == "long":
            return (exit_px / entry_px) - 1.0
        else:
            return (entry_px / exit_px) - 1.0

    def _side_map(self, signal_side: str) -> str:
        if self.map_resistance_short:
            return "short" if str(signal_side).lower() == "resistance" else "long"
        # fallback: support=long, resistance=long (원하면 바꾸세요)
        return "long"

    # ---------- Public: one cycle ----------
    def run_once(self):
        signals = self._read_signals()
        if not signals:
            self._log("no signals")
            return

        # 신규 시그널 → 포지션 생성
        new_count = 0
        for s in signals:
            key = (s.symbol, str(s.ts.value), s.event, s.side)
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)

            # 각 만기별로 독립 포지션 생성
            for ex_h in self.expiries_h:
                pos_id = f"{s.symbol}-{int(s.ts.timestamp())}-{s.event}-{s.side}-{ex_h}"
                if pos_id in self.open_positions:
                    continue
                ohlcv = self._load_ohlcv(s.symbol)
                if ohlcv is None or ohlcv.empty:
                    self._log(f"[SKIP] no OHLCV {s.symbol}")
                    continue
                i_entry, ts_entry, px_entry = self._entry_from_signal(ohlcv, s.ts)
                if i_entry is None:
                    self._log(f"[SKIP] entry not found {s.symbol} {s.ts}")
                    continue

                side = self._side_map(s.side)
                pos = Position(
                    id=pos_id,
                    symbol=s.symbol,
                    side=side,
                    event=s.event,
                    ts_sig=s.ts,
                    ts_entry=ts_entry,
                    entry_px=px_entry,
                    tp_pct=self.tp_pct,
                    sl_pct=self.sl_pct,
                    fee_roundtrip=self.fee_roundtrip,
                    expiry_h=ex_h,
                )
                self.open_positions[pos_id] = pos
                self._append_csv(self.fp_orders_open, [asdict(pos)])
                self._log(f"[OPEN] {pos_id} entry={px_entry:.6f} at {ts_entry}")
                new_count += 1
                if new_count >= self.max_new_per_cycle:
                    break

        # 오픈 포지션 종료 평가
        closed_rows = []
        fill_rows = []
        to_del = []
        for pid, pos in list(self.open_positions.items()):
            ohlcv = self._load_ohlcv(pos.symbol)
            if ohlcv is None or ohlcv.empty:
                continue
            # entry 바가 데이터에 존재하는지 보정
            i_entry = int(pd.DatetimeIndex(ohlcv["ts"]).searchsorted(pos.ts_entry, side="left"))
            if i_entry >= len(ohlcv) or ohlcv["ts"].iloc[i_entry] != pos.ts_entry:
                # 못 찾으면 skip (데이터 동기화 문제)
                continue

            ts_exit, px_exit, reason = self._scan_exit(
                ohlcv=ohlcv, i_entry=i_entry, entry_px=pos.entry_px,
                ts_entry=pos.ts_entry, side=pos.side, expiry_h=pos.expiry_h
            )
            # 아직 해당 구간이 안 닫혔다면 skip
            if ts_exit <= pos.ts_entry:
                continue

            gross = self._gross_ret(pos.entry_px, px_exit, pos.side)
            net = gross - pos.fee_roundtrip

            pos.status = "closed"
            pos.reason = reason
            pos.exit_px = px_exit
            pos.ts_exit = ts_exit
            pos.gross_ret = gross
            pos.net_ret = net

            closed_rows.append(asdict(pos))
            fill_rows.append({
                "ts": ts_exit,
                "id": pid,
                "symbol": pos.symbol,
                "reason": reason,
                "entry_px": pos.entry_px,
                "exit_px": px_exit,
                "gross_ret": gross,
                "net_ret": net,
            })
            to_del.append(pid)
            self._log(f"[CLOSE] {pid} reason={reason} net={net:+.4%}")

        if closed_rows:
            self._append_csv(self.fp_orders_closed, closed_rows)
            self._append_csv(self.fp_fills, fill_rows)
            # equity 업데이트
            eq = pd.read_csv(self.fp_equity)
            equity = float(eq["equity"].iloc[-1]) if not eq.empty else 1.0
            for r in closed_rows:
                equity *= (1.0 + float(r["net_ret"]))
            self._append_csv(self.fp_equity, [{"ts": now_utc(), "equity": equity}])

        for pid in to_del:
            self.open_positions.pop(pid, None)

    # ---------- loop ----------
    def run_loop(self):
        self._log("START loop")
        try:
            while True:
                self.run_once()
                time.sleep(self.poll_secs)
        except KeyboardInterrupt:
            self._log("STOP by user")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("paper_trader")
    p.add_argument("--signals", required=True, help="signals_tv.csv 혹은 가공된 시그널 CSV")
    p.add_argument("--outdir", default=None, help="출력 디렉토리 (기본: logs/realtime/YYYY-MM-DD)")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--ohlcv-roots", default=".;./data;./data/ohlcv;./ohlcv;./logs;./logs/ohlcv")
    p.add_argument("--ohlcv-patterns", default="data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv")
    p.add_argument("--entry", default="prev_close", choices=["prev_close", "next_open"])
    p.add_argument("--tp", type=float, default=1.75, help="익절(%) ex) 1.75")
    p.add_argument("--sl", type=float, default=0.7, help="손절(%) ex) 0.7")
    p.add_argument("--fee", type=float, default=0.001, help="왕복 수수료 (비율) ex) 0.001 = 0.1%")
    p.add_argument("--expiries", default="0.5,1,2", help="만기 시간(시간 단위, 콤마구분)")
    p.add_argument("--poll", type=int, default=10, help="폴링 주기(초)")
    p.add_argument("--max-new", type=int, default=50)
    p.add_argument("--map-res-short", action="store_true", default=True,
                   help="resistance=SHORT, support=LONG 매핑 사용")
    p.add_argument("--dist-max", type=float, default=None, help="distance_pct 필터(비율). 없으면 스킵.")
    p.add_argument("--once", action="store_true", help="한 번만 실행하고 종료")
    args = p.parse_args()

    date_tag = dt.datetime.utcnow().strftime("%Y-%m-%d")
    outdir = args.outdir or os.path.join("logs", "realtime", date_tag)

    expiries = []
    for s in str(args.expiries).split(","):
        s = s.strip().lower().replace("h", "")
        if not s:
            continue
        expiries.append(float(s))

    return {
        "signals_csv": args.signals,
        "outdir": outdir,
        "timeframe": args.timeframe,
        "ohlcv_roots": [x.strip() for x in args.ohlcv_roots.split(";") if x.strip()],
        "ohlcv_patterns": [x.strip() for x in args.ohlcv_patterns.split(";") if x.strip()],
        "entry_policy": args.entry,
        "tp_pct": args.tp / 100.0,
        "sl_pct": args.sl / 100.0,
        "fee_roundtrip": args.fee,
        "expiries_h": expiries,
        "poll_secs": args.poll,
        "max_new_per_cycle": args.max_new,
        "map_resistance_short": args.map_res_short,
        "dist_max": args.dist_max,
        "once": args.once,
    }


def main():
    cfg = parse_args()
    ensure_dir(cfg["outdir"])
    eng = PaperEngine(
        outdir=cfg["outdir"],
        signals_csv=cfg["signals_csv"],
        timeframe=cfg["timeframe"],
        ohlcv_roots=cfg["ohlcv_roots"],
        ohlcv_patterns=cfg["ohlcv_patterns"],
        entry_policy=cfg["entry_policy"],
        tp_pct=cfg["tp_pct"],
        sl_pct=cfg["sl_pct"],
        fee_roundtrip=cfg["fee_roundtrip"],
        expiries_h=cfg["expiries_h"],
        poll_secs=cfg["poll_secs"],
        max_new_per_cycle=cfg["max_new_per_cycle"],
        map_resistance_short=cfg["map_resistance_short"],
        dist_max=cfg["dist_max"],
    )
    if cfg["once"]:
        eng.run_once()
    else:
        eng.run_loop()


if __name__ == "__main__":
    main()