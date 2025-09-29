#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper Trading Engine (patched)
- 백테스트 가정과 최대한 동일한 제약/필터 이식
- 신호 CSV 연동 + 최근신호 윈도우 + 백로그 가드(last_processed_ts)
- 심볼 쿨다운, 틱당 오픈 상한, 동시보유 상한
- long-only 옵션
- 수수료 반영(pnl_net = pnl_gross - 2*fee)
- 로그/CSV 경로: ./logs/paper/
    - trades_open.csv  : 오픈 포지션
    - trades.csv       : 체결/청산 기록
    - equity.csv       : (자리만 확보) 향후 에쿼티 추적용
    - state.json       : last_processed_ts, cooldown 기록
신호 CSV 예상 컬럼(유연 파싱): ts, symbol, event, distance_pct, price
- ts: "YYYY-MM-DD HH:MM:SS" (UTC 가정) 또는 epoch(sec/ms)
- price가 없으면 entry_price=100.0 더미
"""

import os
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

import pandas as pd
import random

# ------------------------- utils -------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def ts_to_str_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def parse_ts_any(x) -> Optional[datetime]:
    """다양한 포맷의 ts를 UTC-aware 로 파싱."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # epoch sec 혹은 ms 추정
        if x > 1e12:  # ms
            return datetime.fromtimestamp(x / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(x, tz=timezone.utc)
    s = str(x).strip()
    if not s:
        return None
    # 1) ISO/스페이스 포맷 시도
    for fmt in ("%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    # 2) pandas 파서로 최후 시도
    try:
        dt = pd.to_datetime(s, utc=True)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.to_pydatetime()
    except Exception:
        return None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ------------------------- data models -------------------------
@dataclass
class OpenTrade:
    id: str
    strategy: str
    expiry_h: float
    side: str
    symbol: str
    event: str
    entry_ts: str   # UTC str
    entry_price: float
    expiry_sec: int
    distance_pct: float

@dataclass
class ClosedTrade:
    id: str
    strategy: str
    expiry_h: float
    side: str
    symbol: str
    event: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    pnl_gross: float      # 비율(+0.008 = +0.8%)
    fee_round: float      # 왕복 수수료 비율(2*fee)
    pnl_net: float        # pnl_gross - fee_round
    status: str           # "expired"

# ------------------------- engine -------------------------
class PaperEngine:
    def __init__(
        self,
        root: str,
        expiries_h: List[float],
        signals_csv: Optional[str] = None,
        long_only: bool = True,
        signals_recent_sec: int = 60,
        min_distance_pct: Optional[float] = None,
        max_distance_pct: Optional[float] = None,
        allow_events: Optional[List[str]] = None,
        cooldown_sec: int = 300,
        max_opens_per_tick: int = 3,
        max_open_positions: int = 30,
        fee: float = 0.001,
        quick_expiry_secs: Optional[int] = None,
        interval: int = 10,
    ):
        self.root = root
        self.expiries_h = expiries_h
        self.signals_csv = signals_csv
        self.long_only = long_only
        self.signals_recent_sec = signals_recent_sec
        self.min_distance_pct = min_distance_pct
        self.max_distance_pct = max_distance_pct
        self.allow_events = set([e.strip() for e in allow_events]) if allow_events else None
        self.cooldown_sec = cooldown_sec
        self.max_opens_per_tick = max_opens_per_tick
        self.max_open_positions = max_open_positions
        self.fee = fee
        self.quick_expiry_secs = quick_expiry_secs
        self.interval = interval

        # paths
        self.logdir = os.path.join(root, "logs", "paper")
        ensure_dir(self.logdir)
        self.fp_trades = os.path.join(self.logdir, "trades.csv")
        self.fp_open = os.path.join(self.logdir, "trades_open.csv")
        self.fp_equity = os.path.join(self.logdir, "equity.csv")
        self.fp_state = os.path.join(self.logdir, "state.json")

        # files bootstrap
        if not os.path.exists(self.fp_equity):
            pd.DataFrame([{"ts": ts_to_str_utc(now_utc()), "equity": 1.0}]).to_csv(self.fp_equity, index=False)

        if not os.path.exists(self.fp_trades):
            pd.DataFrame(columns=[
                "id","strategy","expiry_h","side","symbol","event",
                "entry_ts","exit_ts","entry_price","exit_price",
                "pnl_gross","fee_round","pnl_net","status"
            ]).to_csv(self.fp_trades, index=False)

        if not os.path.exists(self.fp_open):
            pd.DataFrame(columns=[
                "id","strategy","expiry_h","side","symbol","event",
                "entry_ts","entry_price","expiry_sec","distance_pct"
            ]).to_csv(self.fp_open, index=False)

        # state (last_processed_ts, cooldown table)
        self.state = self._load_state()
        self.last_open_time: Dict[str, datetime] = {
            # symbol -> datetime(UTC)
        }
        # 기존 오픈 CSV의 최신 엔트리시각을 쿨다운 힌트로 사용할 수도 있지만 간단화

        logging.info(
            "PaperEngine initialized "
            f"expiries={self.expiries_h}, long_only={self.long_only}, signals_csv={self.signals_csv}, "
            f"recent={self.signals_recent_sec}s, dist=({self.min_distance_pct},{self.max_distance_pct}), "
            f"allow_events={sorted(list(self.allow_events)) if self.allow_events else None}, "
            f"cooldown={self.cooldown_sec}s, max_per_tick={self.max_opens_per_tick}, "
            f"max_open_positions={self.max_open_positions}, fee={self.fee}, quick={self.quick_expiry_secs}"
        )

    # ---------- state io ----------
    def _load_state(self) -> dict:
        if os.path.exists(self.fp_state):
            try:
                with open(self.fp_state, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    # normalize
                    if "last_processed_ts" in obj and obj["last_processed_ts"]:
                        dt = parse_ts_any(obj["last_processed_ts"])
                        obj["last_processed_ts"] = ts_to_str_utc(dt) if dt else None
                    return obj
            except Exception:
                pass
        return {"last_processed_ts": None, "cooldown": {}}

    def _save_state(self):
        try:
            with open(self.fp_state, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"state save failed: {e}")

    # ---------- open/close csv helpers ----------
    def _read_open(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.fp_open)
        except Exception:
            return pd.DataFrame(columns=[
                "id","strategy","expiry_h","side","symbol","event",
                "entry_ts","entry_price","expiry_sec","distance_pct"
            ])

    def _write_open(self, df: pd.DataFrame):
        df.to_csv(self.fp_open, index=False)

    def _append_closed(self, rows: List[ClosedTrade]):
        if not rows:
            return
        df = pd.DataFrame([asdict(r) for r in rows])
        df.to_csv(self.fp_trades, mode="a", header=not os.path.getsize(self.fp_trades), index=False)

    # ---------- signals loader & filters ----------
    def _load_signals_df(self) -> pd.DataFrame:
        """신호 CSV를 관대한 스키마로 로드하고 핵심 컬럼을 빚어낸다."""
        if not self.signals_csv or not os.path.exists(self.signals_csv):
            return pd.DataFrame(columns=["ts","symbol","event","distance_pct","price"])
        try:
            df = pd.read_csv(self.signals_csv)
        except UnicodeDecodeError:
            df = pd.read_csv(self.signals_csv, encoding="cp949")
        except Exception:
            df = pd.read_csv(self.signals_csv, engine="python")
        cols = {c.lower(): c for c in df.columns}

        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_ts = pick("ts","timestamp","time","datetime")
        c_sym = pick("symbol","ticker","market")
        c_evt = pick("event","signal","type")
        c_dist = pick("distance_pct","distance","dist_pct","dist")
        c_price = pick("price","close","entry_price")

        # 최소 ts/symbol/event는 필요
        if not c_ts or not c_sym or not c_evt:
            return pd.DataFrame(columns=["ts","symbol","event","distance_pct","price"])

        out = pd.DataFrame({
            "ts": df[c_ts],
            "symbol": df[c_sym],
            "event": df[c_evt],
        })
        out["distance_pct"] = df[c_dist] if c_dist else None
        out["price"] = df[c_price] if c_price else None

        # ts 파싱
        out["ts_dt"] = out["ts"].apply(parse_ts_any)
        out = out[out["ts_dt"].notna()].copy()
        # UTC 정렬
        out.sort_values("ts_dt", inplace=True)
        return out

    def _filter_signals(self, sig_df: pd.DataFrame) -> pd.DataFrame:
        if sig_df.empty:
            return sig_df

        nowdt = now_utc()
        # 최근 N초 윈도우
        if self.signals_recent_sec and self.signals_recent_sec > 0:
            cutoff = nowdt - timedelta(seconds=self.signals_recent_sec)
            sig_df = sig_df[sig_df["ts_dt"] >= cutoff]

        # 백로그 가드: state.last_processed_ts 이후만
        if self.state.get("last_processed_ts"):
            last_dt = parse_ts_any(self.state["last_processed_ts"])
            if last_dt:
                sig_df = sig_df[sig_df["ts_dt"] > last_dt]

        # 이벤트 화이트리스트
        if self.allow_events:
            sig_df = sig_df[sig_df["event"].astype(str).str.lower().isin(self.allow_events)]

        # 거리 필터
        if self.min_distance_pct is not None:
            sig_df = sig_df[(sig_df["distance_pct"].astype(float) >= self.min_distance_pct)]
        if self.max_distance_pct is not None:
            sig_df = sig_df[(sig_df["distance_pct"].astype(float) <= self.max_distance_pct)]

        # 심볼 쿨다운 (state.cooldown 에 마지막 오픈시각 보관)
        cd = self.state.get("cooldown", {})
        keep_rows = []
        for _, r in sig_df.iterrows():
            sym = str(r["symbol"])
            last_open_iso = cd.get(sym)
            ok = True
            if last_open_iso:
                last_open_dt = parse_ts_any(last_open_iso)
                if last_open_dt and (nowdt - last_open_dt).total_seconds() < self.cooldown_sec:
                    ok = False
            if ok:
                keep_rows.append(r)
        sig_df = pd.DataFrame(keep_rows) if keep_rows else pd.DataFrame(columns=sig_df.columns)

        return sig_df

    # ---------- open trades from signals ----------
    def _open_from_signals(self):
        if not self.signals_csv:
            return 0

        raw = self._load_signals_df()
        if raw.empty:
            return 0
        cand = self._filter_signals(raw)
        if cand.empty:
            return 0

        # 동시보유 상한 확인
        open_df = self._read_open()
        current_open = 0 if open_df.empty else len(open_df)
        capacity = max(self.max_open_positions - current_open, 0)
        if capacity <= 0:
            return 0

        # 틱당 오픈 상한 & 동시보유 cap 동시 적용
        budget = min(self.max_opens_per_tick, capacity)

        opened = 0
        rows = []
        # 최신 신호 우선 (가장 최근부터)
        cand = cand.sort_values("ts_dt", ascending=True)  # 오래된→새로운 순으로 훑고 마지막 처리 ts 업데이트

        for _, r in cand.iterrows():
            if opened >= budget:
                break
            sym = str(r["symbol"])
            evt = str(r["event"]).lower()
            # side: long-only면 long만
            side = "long" if self.long_only else "long"  # 현재는 long만 사용
            entry_price = float(r["price"]) if pd.notna(r["price"]) else 100.0
            dist = float(r["distance_pct"]) if pd.notna(r["distance_pct"]) else 0.0

            for eh in self.expiries_h:
                if opened >= budget:
                    break
                expiry_sec = int(self.quick_expiry_secs if self.quick_expiry_secs else eh * 3600)
                trade_id = f"T{int(time.time()*1000)}_{sym}_{str(eh).replace('.','_')}"
                row = OpenTrade(
                    id=trade_id,
                    strategy="paper_live",
                    expiry_h=eh,
                    side=side,
                    symbol=sym,
                    event=evt,
                    entry_ts=ts_to_str_utc(r["ts_dt"]),
                    entry_price=entry_price,
                    expiry_sec=expiry_sec,
                    distance_pct=dist,
                )
                rows.append(asdict(row))
                opened += 1

                # cooldown 갱신
                self.state.setdefault("cooldown", {})[sym] = ts_to_str_utc(now_utc())

                if opened >= budget:
                    break

        if rows:
            new_df = pd.DataFrame(rows)
            if open_df.empty:
                out_df = new_df
            else:
                out_df = pd.concat([open_df, new_df], ignore_index=True)
            self._write_open(out_df)

        # 처리한 신호의 최종 ts를 state에 저장(백로그 가드)
        last_ts = cand["ts_dt"].max()
        if last_ts is not None:
            self.state["last_processed_ts"] = ts_to_str_utc(last_ts)
            self._save_state()

        if opened:
            logging.info(f"opened {opened} trade(s) from signals")
        return opened

    # ---------- close expired ----------
    def _close_expired(self):
        open_df = self._read_open()
        if open_df.empty:
            return 0

        nowdt = now_utc()
        remaining = []
        closed_rows: List[ClosedTrade] = []

        for _, r in open_df.iterrows():
            entry_dt = parse_ts_any(r["entry_ts"])
            expiry_td = timedelta(seconds=int(r["expiry_sec"]))
            if entry_dt and nowdt >= entry_dt + expiry_td:
                # 더미 가격변화: 만기 짧을수록 분산↑ (0.5h ~ 2h 백테 감성 반영 살짝)
                eh = float(r["expiry_h"])
                base = 0.0025 if eh <= 0.5 else (0.0020 if eh <= 1.0 else 0.0016)
                drift = random.uniform(0.5 * base, 1.2 * base)
                sign = 1 if random.random() < 0.6 else -1  # 승률 60% 근방
                pnl_gross = sign * drift

                fee_round = 2.0 * self.fee
                pnl_net = pnl_gross - fee_round
                entry_price = float(r["entry_price"])
                exit_price = entry_price * (1.0 + pnl_gross)

                closed = ClosedTrade(
                    id=r["id"],
                    strategy=str(r["strategy"]),
                    expiry_h=float(r["expiry_h"]),
                    side=str(r["side"]),
                    symbol=str(r["symbol"]),
                    event=str(r["event"]),
                    entry_ts=str(r["entry_ts"]),
                    exit_ts=ts_to_str_utc(nowdt),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_gross=pnl_gross,
                    fee_round=fee_round,
                    pnl_net=pnl_net,
                    status="expired",
                )
                closed_rows.append(closed)
            else:
                remaining.append(r)

        # 기록 갱신
        n_closed = 0
        if closed_rows:
            self._append_closed(closed_rows)
            n_closed = len(closed_rows)
            logging.info(f"closed {n_closed} trade(s)")

        if remaining:
            self._write_open(pd.DataFrame(remaining))
        else:
            # 모두 닫혔으면 빈 프레임으로 초기화
            self._write_open(pd.DataFrame(columns=[
                "id","strategy","expiry_h","side","symbol","event",
                "entry_ts","entry_price","expiry_sec","distance_pct"
            ]))
        return n_closed

    # ---------- main ticks ----------
    def run_once(self):
        # 1) 신호 기반 신규 오픈 (필터/제약 적용, 백로그 가드)
        self._open_from_signals()
        # 2) 만기 도달한 포지션 종료
        self._close_expired()

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

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # 모드
    p.add_argument("--once", action="store_true", help="Run one tick and exit")
    p.add_argument("--run-for", type=int, default=None, help="Run for N minutes")
    p.add_argument("--interval", type=int, default=10, help="Loop interval seconds")

    # 만기
    p.add_argument("--expiries", type=str, default="0.5,1,2", help="Expiry hours list, e.g. '0.5,1,2'")
    p.add_argument("--quick-expiry-secs", type=int, default=None, help="Quick test expiry seconds (e.g. 36)")

    # 신호
    p.add_argument("--signals-csv", type=str, default=None, help="Path to signals csv")
    p.add_argument("--long-only", action="store_true", help="Open only long positions (recommended)")

    # 백테 조건 이식
    p.add_argument("--signals-recent-sec", type=int, default=60, help="Only accept signals within last N seconds")
    p.add_argument("--min-distance-pct", type=float, default=None)
    p.add_argument("--max-distance-pct", type=float, default=None)
    p.add_argument("--allow-events", type=str, default=None, help="Comma list, e.g. 'box_breakout,line_breakout'")
    p.add_argument("--cooldown-sec", type=int, default=300)
    p.add_argument("--max-opens-per-tick", type=int, default=3)
    p.add_argument("--max-open-positions", type=int, default=30)

    # 수수료/리스크(더미 체결에 수수료 반영)
    p.add_argument("--fee", type=float, default=0.001, help="Per-side fee (0.001 = 0.1%)")
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

    # expiries
    if args.quick_expiry_secs and args.quick_expiry_secs > 0:
        expiries_h = [args.quick_expiry_secs / 3600.0]
        note = f"(quick={args.quick_expiry_secs}s)"
    else:
        expiries_h = [float(x) for x in str(args.expiries).split(",") if x.strip()]
        note = ""

    # events whitelist
    allow_events = None
    if args.allow_events:
        allow_events = [x.strip().lower() for x in args.allow_events.split(",") if x.strip()]

    logging.info(f"expiries_h={expiries_h} {note}")
    logging.info(f"long_only={bool(args.long_only)}")

    eng = PaperEngine(
        root=root,
        expiries_h=expiries_h,
        signals_csv=args.signals_csv,
        long_only=bool(args.long_only),
        signals_recent_sec=args.signals_recent_sec,
        min_distance_pct=args.min_distance_pct,
        max_distance_pct=args.max_distance_pct,
        allow_events=allow_events,
        cooldown_sec=args.cooldown_sec,
        max_opens_per_tick=args.max_opens_per_tick,
        max_open_positions=args.max_open_positions,
        fee=args.fee,
        quick_expiry_secs=args.quick_expiry_secs,
        interval=args.interval,
    )

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