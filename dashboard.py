#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight live dashboard for paper_trader
- Reads CSVs under ./logs/paper and ./logs
- Aggregates counters (opened/closed, rejects by reason, recent signals, etc.)
- Serves a small HTML page that polls /api/snapshot every 5s

Run:
  (.venv) python dashboard.py --port 8090
Open:
  http://127.0.0.1:8090
"""
import os, time, argparse, json, math
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, Response

import pandas as pd

UTC = timezone.utc
ROOT = os.path.abspath(os.path.dirname(__file__))
LOGS = os.path.join(ROOT, "logs")
PAPER = os.path.join(LOGS, "paper")

PATH_REJECTS = os.path.join(PAPER, "rejects.csv")
PATH_OPEN    = os.path.join(PAPER, "trades_open.csv")
PATH_CLOSED  = os.path.join(PAPER, "trades_closed.csv")  # 있으면 사용
PATH_ENGINE_START = os.path.join(PAPER, "engine_start.txt")
PATH_SIGNALS = os.path.join(LOGS, "signals_tv.csv")

ALLOWED_EVENTS = {"line_breakout", "box_breakout"}

# ---------- utils ----------
def _dt_utcnow():
    return datetime.now(tz=UTC)

def _safe_mtime(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), tz=UTC).isoformat()
    except Exception:
        return None

def _read_csv(path, cols=None):
    """
    Read CSV safely with pandas.
    If file missing or empty, returns empty DataFrame with requested columns.
    """
    if cols is None:
        cols = []
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
        # Ensure requested columns exist
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df
    except Exception as e:
        print(f"[DASH][WARN] failed to read {path}: {e}")
        return pd.DataFrame(columns=cols)

def _parse_iso(s):
    if pd.isna(s) or s is None or str(s).strip()=="":
        return None
    try:
        # pandas.to_datetime handles many formats incl. timezone
        return pd.to_datetime(s, utc=True).to_pydatetime()
    except Exception:
        try:
            return datetime.fromisoformat(str(s))
        except Exception:
            return None

def _read_engine_start():
    if not os.path.exists(PATH_ENGINE_START):
        return None
    try:
        txt = open(PATH_ENGINE_START, "r", encoding="utf-8").read().strip()
        return _parse_iso(txt)
    except Exception:
        return None

# ---------- snapshot builder ----------
def build_snapshot(recent_min=10):
    now = _dt_utcnow()
    engine_start = _read_engine_start()

    # trades_open
    open_cols = ["opened_at","symbol","event","side","level","expire_at","fee"]
    df_open = _read_csv(PATH_OPEN, open_cols)
    n_open_positions = len(df_open)

    # trades_closed (optional)
    closed_cols = ["opened_at","closed_at","symbol","event","side","level","pnl","fee"]
    df_closed = _read_csv(PATH_CLOSED, closed_cols)
    n_closed = len(df_closed)

    # rejects
    rej_cols = ["ts","symbol","event","side","level","distance_pct","phase","reason","engine_start","now"]
    df_rej = _read_csv(PATH_REJECTS, rej_cols)
    # normalize timestamps
    if "now" in df_rej.columns:
        df_rej["now_dt"] = df_rej["now"].apply(_parse_iso)
    else:
        df_rej["now_dt"] = None
    if "ts" in df_rej.columns:
        df_rej["ts_dt"] = df_rej["ts"].apply(_parse_iso)
    else:
        df_rej["ts_dt"] = None

    # total rejects (all)
    total_rejects = len(df_rej)

    # rejects by reason (all)
    by_reason = {}
    if "reason" in df_rej.columns:
        tmp = df_rej["reason"].value_counts(dropna=False)
        for k, v in tmp.items():
            by_reason[str(k)] = int(v)

    # rejects since engine_start (phase=='run' preferred)
    run_mask = None
    if engine_start is not None and "now_dt" in df_rej.columns:
        run_mask = df_rej["now_dt"].notna() & (df_rej["now_dt"] >= engine_start)
    else:
        run_mask = df_rej["now_dt"].notna()

    df_rej_run = df_rej[run_mask].copy() if run_mask is not None else df_rej.head(0)
    if "phase" in df_rej_run.columns:
        df_rej_run = df_rej_run[df_rej_run["phase"] == "run"]

    recent_td = timedelta(minutes=recent_min)
    df_rej_recent = df_rej[df_rej["now_dt"].notna() & (df_rej["now_dt"] >= (now - recent_td))]
    recent_rejects_total = len(df_rej_recent)
    recent_rejects_reason = {}
    if not df_rej_recent.empty and "reason" in df_rej_recent.columns:
        tmp2 = df_rej_recent["reason"].value_counts(dropna=False)
        for k, v in tmp2.items():
            recent_rejects_reason[str(k)] = int(v)

    # signals (recent, allowed)
    sig_cols = ["ts","event","side","level","symbol","distance_pct"]
    df_sig = _read_csv(PATH_SIGNALS, sig_cols)
    if "ts" in df_sig.columns:
        df_sig["ts_dt"] = df_sig["ts"].apply(_parse_iso)
    else:
        df_sig["ts_dt"] = None

    recent_signals = 0
    recent_allowed = 0
    if not df_sig.empty:
        m_recent = df_sig["ts_dt"].notna() & (df_sig["ts_dt"] >= (now - recent_td))
        df_sig_recent = df_sig[m_recent]
        recent_signals = len(df_sig_recent)
        if "event" in df_sig_recent.columns:
            recent_allowed = int((df_sig_recent["event"].astype(str).isin(ALLOWED_EVENTS)).sum())

    # opened (count) from open book (this file is “current open positions” so it’s not a counter)
    # instead, derive opened-since-start from rejects? Not accurate.
    # Better: show open book rows + closed rows as simple numbers.
    n_opened_total = len(df_open)  # “current open positions”
    n_closed_total = n_closed

    snapshot = {
        "engine": {
            "engine_start": engine_start.isoformat() if engine_start else None,
            "now": now.isoformat(),
            "uptime_min": None if not engine_start else int((now - engine_start).total_seconds() // 60),
        },
        "files": {
            "rejects": {"path": PATH_REJECTS, "mtime": _safe_mtime(PATH_REJECTS), "rows": int(total_rejects)},
            "trades_open": {"path": PATH_OPEN, "mtime": _safe_mtime(PATH_OPEN), "rows": int(n_open_positions)},
            "trades_closed": {"path": PATH_CLOSED, "mtime": _safe_mtime(PATH_CLOSED), "rows": int(n_closed)},
            "signals_tv": {"path": PATH_SIGNALS, "mtime": _safe_mtime(PATH_SIGNALS), "rows": int(len(df_sig))},
        },
        "positions": {
            "open_positions": int(n_open_positions),
            "closed_trades": int(n_closed_total),
        },
        "rejects": {
            "total": int(total_rejects),
            "by_reason": by_reason,
            "run_phase_since_engine_start": int(len(df_rej_run)) if engine_start else None,
            "recent_window_min": int(recent_min),
            "recent_total": int(recent_rejects_total),
            "recent_by_reason": recent_rejects_reason,
        },
        "signals": {
            "recent_window_min": int(recent_min),
            "recent_total": int(recent_signals),
            "recent_allowed": int(recent_allowed),
        },
    }
    return snapshot

# ---------- web app ----------
app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Paper Trader Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .card { padding: 14px; border: 1px solid #e5e7eb; border-radius: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .muted { color:#6b7280; font-size: 12px }
    h1 { font-size: 20px; margin: 0 0 12px }
    h2 { font-size: 14px; margin: 0 0 8px }
    .kvs { margin: 4px 0; }
    .kvs div { display:flex; justify-content: space-between; margin:2px 0; }
    code { background:#f3f4f6; padding:2px 6px; border-radius:6px; }
    .byreason { font-size: 13px; line-height: 1.3; }
  </style>
</head>
<body>
  <h1>Paper Trader Dashboard</h1>
  <div class="muted" id="meta"></div>

  <div class="grid">
    <div class="card">
      <h2>Engine</h2>
      <div class="kvs" id="engine"></div>
    </div>
    <div class="card">
      <h2>Positions</h2>
      <div class="kvs" id="positions"></div>
    </div>
    <div class="card">
      <h2>Signals (recent)</h2>
      <div class="kvs" id="signals"></div>
    </div>
    <div class="card">
      <h2>Rejects (total)</h2>
      <div class="kvs" id="rejects_total"></div>
      <div class="byreason" id="rejects_byreason"></div>
    </div>
    <div class="card">
      <h2>Rejects (recent)</h2>
      <div class="kvs" id="rejects_recent"></div>
      <div class="byreason" id="rejects_recent_byreason"></div>
    </div>
    <div class="card">
      <h2>Files</h2>
      <div class="kvs" id="files"></div>
    </div>
  </div>

  <p class="muted" style="margin-top:12px">Auto refresh: <code>5s</code></p>

<script>
async function fetchSnapshot() {
  const res = await fetch('/api/snapshot');
  const j = await res.json();

  // meta
  const meta = document.getElementById('meta');
  meta.textContent = `Updated: ${new Date(j.engine.now).toLocaleString()} | Recent Window: ${j.signals.recent_window_min} min`;

  // engine
  const e = j.engine;
  document.getElementById('engine').innerHTML = `
    <div><span>Engine start</span><strong>${e.engine_start ?? '-'}</strong></div>
    <div><span>Uptime (min)</span><strong>${e.uptime_min ?? '-'}</strong></div>
  `;

  // positions
  const p = j.positions;
  document.getElementById('positions').innerHTML = `
    <div><span>Open positions</span><strong>${p.open_positions}</strong></div>
    <div><span>Closed trades</span><strong>${p.closed_trades}</strong></div>
  `;

  // signals
  const s = j.signals;
  document.getElementById('signals').innerHTML = `
    <div><span>Recent total</span><strong>${s.recent_total}</strong></div>
    <div><span>Recent allowed</span><strong>${s.recent_allowed}</strong></div>
  `;

  // rejects total
  const r = j.rejects;
  document.getElementById('rejects_total').innerHTML = `
    <div><span>Total</span><strong>${r.total}</strong></div>
    <div><span>Run-phase since start</span><strong>${r.run_phase_since_engine_start ?? '-'}</strong></div>
  `;
  const br = r.by_reason || {};
  document.getElementById('rejects_byreason').innerHTML =
    Object.keys(br).length ? Object.entries(br).map(([k,v])=>`${k}: <strong>${v}</strong>`).join('<br>') : '<em>-</em>';

  // rejects recent
  document.getElementById('rejects_recent').innerHTML = `
    <div><span>Recent total</span><strong>${r.recent_total}</strong></div>
  `;
  const brr = r.recent_by_reason || {};
  document.getElementById('rejects_recent_byreason').innerHTML =
    Object.keys(brr).length ? Object.entries(brr).map(([k,v])=>`${k}: <strong>${v}</strong>`).join('<br>') : '<em>-</em>';

  // files
  const f = j.files;
  document.getElementById('files').innerHTML = `
    <div><span>rejects.csv</span><span><strong>${f.rejects.rows}</strong> rows<br><span class="muted">${f.rejects.mtime ?? '-'}</span></span></div>
    <div><span>trades_open.csv</span><span><strong>${f.trades_open.rows}</strong> rows<br><span class="muted">${f.trades_open.mtime ?? '-'}</span></span></div>
    <div><span>trades_closed.csv</span><span><strong>${f.trades_closed.rows}</strong> rows<br><span class="muted">${f.trades_closed.mtime ?? '-'}</span></span></div>
    <div><span>signals_tv.csv</span><span><strong>${f.signals_tv.rows}</strong> rows<br><span class="muted">${f.signals_tv.mtime ?? '-'}</span></span></div>
  `;
}

fetchSnapshot();
setInterval(fetchSnapshot, 5000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")

@app.route("/api/snapshot")
def api_snapshot():
    try:
        snap = build_snapshot(recent_min=10)
        return jsonify(snap)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    print(f"[DASH] serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()