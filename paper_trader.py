#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import time
import argparse
import logging
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import random
from typing import Optional, List, Tuple

# ------------------------- utils -------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def ts_str(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_ts_any(s: str) -> datetime:
    """
    문자열 타임스탬프를 UTC-aware datetime으로 파싱.
    지원: 'YYYY-mm-dd HH:MM:SS' (naive→UTC 가정), ISO8601(+offset 포함)
    """
    s = str(s).strip()
    # ISO8601 시도
    try:
        dt = pd.to_datetime(s, utc=True)
        if pd.isna(dt):
            raise ValueError
        return dt.to_pydatetime()
    except Exception:
        pass
    # naive 포맷(기존 요약 통계와 일관)
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {s}")

# ------------------------- data models -------------------------
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
    signal_ts: str  # 원본 시그널 시각(트레이싱용)

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
    pnl: float
    status: str  # "expired"

# ------------------------- engine -------------------------
class PaperEngine:
    def __init__(
        self,
        root: str,
        expiries_h: List[float],
        quick_expiry_secs: Optional[int] = None,
        interval: int = 10,
        long_only: bool = True,
        signals_csv: Optional[str] = None,
        default_strategy: str = "tv_signal",
        cooldown_sec: int = 60,
    ):
        self.root = root
        self.expiries_h = expiries_h
        self.quick_expiry_secs = quick_expiry_secs
        self.interval = interval
        self.long_only = long_only
        self.signals_csv = signals_csv or os.path.join(root, "logs", "signals_tv.csv")
        self.default_strategy = default_strategy
        self.cooldown_sec = max(0, int(cooldown_sec))
        self.running = True

        self.logdir = os.path.join(root, "logs", "paper")
        ensure_dir(self.logdir)

        self.fp_trades = os.path.join(self.logdir, "trades.csv")
        self.fp_open = os.path.join(self.logdir, "trades_open.csv")
        self.fp_equity = os.path.join(self.logdir, "equity.csv")

        # equity.csv 헤더 보장
        if not os.path.exists(self.fp_equity):
            pd.DataFrame([{"ts": ts_str(now_utc()), "equity": 1.0}]).to_csv(self.fp_equity, index=False)

        # trades.csv 헤더 보장
        if not os.path.exists(self.fp_trades):
            pd.DataFrame(columns=[
                "id","strategy","symbol","event","expiry_h","side",
                "entry_ts","exit_ts","entry_price","exit_price","pnl","status"
            ]).to_csv(self.fp_trades, index=False)

        # trades_open.csv 헤더 보장
        if not os.path.exists(self.fp_open):
            pd.DataFrame(columns=[
                "id","strategy","symbol","event","expiry_h","side",
                "entry_ts","entry_price","expiry_sec","signal_ts"
            ]).to_csv(self.fp_open, index=False)

        # long-only라면 기존 short 오픈 포지션 제거
        if self.long_only and os.path.exists(self.fp_open):
            try:
                df = pd.read_csv(self.fp_open)
                if not df.empty and "side" in df.columns:
                    new_df = df[df["side"].astype(str).str.lower() == "long"].copy()
                    if len(new_df) != len(df):
                        logging.info(f"purged {len(df) - len(new_df)} short open trades (long-only mode)")
                        new_df.to_csv(self.fp_open, index=False)
            except Exception as e:
                logging.warning(f"cleanup open(short) failed: {e}")

        # signals incremental read offset 파일
        self.fp_sig_offset = self.signals_csv + ".offset"

        logging.info(
            f"PaperEngine initialized expiries={self.expiries_h}, quick={self.quick_expiry_secs}, "
            f"long_only={self.long_only}, signals_csv={self.signals_csv}, cooldown={self.cooldown_sec}s"
        )

    # ---------- IO helpers ----------
    def _read_open(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.fp_open)
        except Exception:
            return pd.DataFrame(columns=[
                "id","strategy","symbol","event","expiry_h","side",
                "entry_ts","entry_price","expiry_sec","signal_ts"
            ])

    def _write_open(self, df: pd.DataFrame):
        df.to_csv(self.fp_open, index=False)

    def _append_closed(self, rows: List[ClosedTrade]):
        if not rows:
            return
        df = pd.DataFrame([asdict(r) for r in rows])
        df.to_csv(self.fp_trades, mode="a", header=not os.path.getsize(self.fp_trades), index=False)

    # ---------- signals incremental reader ----------
    def _read_new_signal_lines(self) -> Optional[str]:
        """signals_csv에서 지난 오프셋 이후의 raw 텍스트를 반환."""
        if not os.path.exists(self.signals_csv):
            return None
        last_off = 0
        if os.path.exists(self.fp_sig_offset):
            try:
                with open(self.fp_sig_offset, "r", encoding="utf-8") as f:
                    last_off = int(f.read().strip() or "0")
            except Exception:
                last_off = 0
        size_now = os.path.getsize(self.signals_csv)
        if size_now <= last_off:
            return None  # 신규 없음
        with open(self.signals_csv, "rb") as f:
            f.seek(last_off)
            chunk = f.read()
        # 오프셋 갱신
        with open(self.fp_sig_offset, "w", encoding="utf-8") as f:
            f.write(str(size_now))
        # CSV 텍스트(추가분)
        try:
            return chunk.decode("utf-8")
        except UnicodeDecodeError:
            return chunk.decode("cp949", errors="replace")

    def _parse_signals_csv_text(self, csv_text: str) -> pd.DataFrame:
        """
        추가된 CSV 텍스트를 DF로 파싱. 최소 컬럼(ts,symbol,event)이 필요.
        선택 컬럼: strategy, side, expiry_h
        """
        if not csv_text:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(csv_text))
        # 표준화
        cols = {c.lower(): c for c in df.columns}
        need = ["ts", "symbol", "event"]
        if not all(x in [k.lower() for k in df.columns] for x in need):
            # 헤더가 포함되지 않은 chunk일 수 있으므로 전체파일 재파싱 폴백
            try:
                full = pd.read_csv(self.signals_csv)
                cols = {c.lower(): c for c in full.columns}
                if not all(k in cols for k in ["ts","symbol","event"]):
                    logging.warning("signals csv missing required columns: ts,symbol,event")
                    return pd.DataFrame()
                return full.tail(1000)  # 최근부 1000행만
            except Exception as e:
                logging.warning(f"parse signals fallback failed: {e}")
                return pd.DataFrame()

        # 최소 컬럼 lower-key로 접근
        def col(name):  # name is lower
            return df[cols[name]] if name in cols else None

        out = pd.DataFrame({
            "ts": col("ts"),
            "symbol": col("symbol"),
            "event": col("event")
        }).copy()

        out["strategy"] = col("strategy") if "strategy" in cols else self.default_strategy
        out["side"] = (col("side").astype(str).str.lower()
                       if "side" in cols else ("long" if self.long_only else "long"))
        # expiry_h는 없으면 None 두고, 오픈 때 엔진 설정(expiries_h 또는 quick)로 확장
        out["expiry_h"] = (pd.to_numeric(col("expiry_h"), errors="coerce")
                           if "expiry_h" in cols else None)

        # ts 파싱/UTC
        try:
            out["signal_dt"] = out["ts"].map(parse_ts_any)
        except Exception as e:
            logging.warning(f"signal ts parse error: {e}")
            out["signal_dt"] = now_utc()

        # 불필요/결측 제거
        out = out.dropna(subset=["ts", "symbol", "event"])
        return out

    # ---------- entry from signals ----------
    def _open_from_signals(self):
        """
        신규 시그널을 읽어 만기 리스트(expiries_h 혹은 quick)별로 포지션 오픈.
        - long_only면 side를 long으로 강제
        - 동일 (symbol,event,expiry_h,strategy) key의 오픈 포지션이 이미 있거나
          최근 쿨다운 내에 같은 key가 생성되었으면 skip
        """
        csv_text = self._read_new_signal_lines()
        if csv_text is None:
            return  # 신규 없음

        sig_df = self._parse_signals_csv_text(csv_text)
        if sig_df.empty:
            return

        open_df = self._read_open()

        # 최근 생성 내역(쿨다운 확인용) - open_df의 entry_ts를 사용
        def recent_exists(key: Tuple[str, str, float, str], ref_dt: datetime) -> bool:
            sym, evt, eh, strat = key
            if open_df.empty:
                return False
            try:
                mask = (
                    (open_df["symbol"] == sym) &
                    (open_df["event"] == evt) &
                    (open_df["strategy"] == strat) &
                    (open_df["expiry_h"].astype(float) == float(eh))
                )
                sub = open_df.loc[mask]
                if sub.empty:
                    return False
                # 쿨다운 확인
                for _, r in sub.iterrows():
                    et = parse_ts_any(str(r["entry_ts"]))
                    if (ref_dt - et).total_seconds() < self.cooldown_sec:
                        return True
                return False
            except Exception:
                return False

        # 준비
        created = 0
        now_s = ts_str(now_utc())
        # 어떤 만기를 적용할지
        expiries = [self.quick_expiry_secs / 3600.0] if (self.quick_expiry_secs and self.quick_expiry_secs > 0) else self.expiries_h

        # 시그널 루프
        for _, s in sig_df.iterrows():
            sym = str(s["symbol"])
            evt = str(s["event"])
            strat = str(s.get("strategy", self.default_strategy))
            side = "long" if self.long_only else str(s.get("side", "long")).lower()
            if self.long_only and side != "long":
                side = "long"

            # 만기 리스트 확정: 시그널에 expiry_h가 있으면 그 값만, 없으면 엔진 만기들 전체
            sig_eh = s.get("expiry_h", None)
            if pd.notna(sig_eh):
                use_expiries = [float(sig_eh)]
            else:
                use_expiries = [float(eh) for eh in expiries]

            for eh in use_expiries:
                key = (sym, evt, float(eh), strat)

                # 이미 같은 만기의 오픈 포지션 존재?
                exists = False
                if not open_df.empty:
                    try:
                        exists = any(
                            (open_df["symbol"] == sym)
                            & (open_df["event"] == evt)
                            & (open_df["strategy"] == strat)
                            & (open_df["expiry_h"].astype(float) == float(eh))
                        )
                    except Exception:
                        exists = False

                if exists or recent_exists(key, s["signal_dt"]):
                    continue  # 중복/쿨다운 skip

                # 새 엔트리 생성
                trade_id = f"S{int(time.time()*1000)}_{sym}_{evt}_{str(eh).replace('.','_')}"
                entry_price = 100.0  # 실제 체결가격 연동은 후속단계(가격피드)에서
                expiry_sec = int(self.quick_expiry_secs if self.quick_expiry_secs else eh * 3600)

                row = OpenTrade(
                    id=trade_id,
                    strategy=strat,
                    symbol=sym,
                    event=evt,
                    expiry_h=float(eh),
                    side=side,
                    entry_ts=now_s,
                    entry_price=entry_price,
                    expiry_sec=expiry_sec,
                    signal_ts=ts_str(s["signal_dt"]),
                )

                new_row = pd.DataFrame([asdict(row)])
                frames = [df for df in (open_df, new_row) if not df.empty]
                open_df = pd.concat(frames, ignore_index=True) if frames else new_row
                created += 1

        if created:
            self._write_open(open_df)
            logging.info(f"opened {created} trade(s) from signals")

    # ---------- close on expiry ----------
    def _close_expired(self):
        open_df = self._read_open()
        if open_df.empty:
            return

        # long-only 보호
        if self.long_only and "side" in open_df.columns:
            before = len(open_df)
            open_df = open_df[open_df["side"].astype(str).str.lower() == "long"].copy()
            purged = before - len(open_df)
            if purged > 0:
                logging.info(f"purged {purged} short open trade(s) before close (long-only mode)")
                self._write_open(open_df)

        now_dt = now_utc()
        remaining = []
        closed_rows: List[ClosedTrade] = []

        for _, r in open_df.iterrows():
            entry_dt = parse_ts_any(str(r["entry_ts"]))
            expiry_td = timedelta(seconds=int(r["expiry_sec"]))
            if now_dt >= entry_dt + expiry_td:
                # 더미 PnL: ±0.1~0.3%
                drift = random.uniform(0.001, 0.003)
                sign = 1 if random.random() < 0.55 else -1
                pnl = sign * drift
                exit_price = float(r["entry_price"]) * (1.0 + pnl)

                closed = ClosedTrade(
                    id=str(r["id"]),
                    strategy=str(r["strategy"]),
                    symbol=str(r["symbol"]),
                    event=str(r["event"]),
                    expiry_h=float(r["expiry_h"]),
                    side=str(r["side"]),
                    entry_ts=str(r["entry_ts"]),
                    exit_ts=ts_str(now_dt),
                    entry_price=float(r["entry_price"]),
                    exit_price=exit_price,
                    pnl=pnl,
                    status="expired",
                )
                closed_rows.append(closed)
            else:
                remaining.append(r)

        if closed_rows:
            self._append_closed(closed_rows)
            logging.info(f"closed {len(closed_rows)} trade(s)")

        if remaining:
            self._write_open(pd.DataFrame(remaining))
        else:
            self._write_open(pd.DataFrame(columns=[
                "id","strategy","symbol","event","expiry_h","side",
                "entry_ts","entry_price","expiry_sec","signal_ts"
            ]))

    # ---------- main ticks ----------
    def run_once(self):
        # 1) 시그널 ingest → open
        self._open_from_signals()
        # 2) 만기 도달 → close
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
    p.add_argument("--allow-short", action="store_true", help="Allow short signals (default: long-only)")
    p.add_argument("--signals-csv", type=str, default=None, help="Path to signals CSV (default: logs/signals_tv.csv)")
    p.add_argument("--strategy-name", type=str, default="tv_signal", help="Default strategy name when CSV lacks 'strategy'")
    p.add_argument("--cooldown-sec", type=int, default=60, help="Cooldown seconds per (symbol,event,expiry,strategy)")
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

    long_only = not args.allow_short
    logging.info(f"expiries_h={expiries_h} {note}")
    logging.info(f"long_only={long_only}")

    eng = PaperEngine(
        root=root,
        expiries_h=expiries_h,
        quick_expiry_secs=args.quick_expiry_secs,
        interval=args.interval,
        long_only=long_only,
        signals_csv=args.signals_csv,
        default_strategy=args.strategy_name,
        cooldown_sec=args.cooldown_sec,
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