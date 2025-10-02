# paper_dashboard.py
# Flask + Chart.js dashboard for Paper Trader (dark, card layout)
# deps: pip install flask pandas

import os, csv, math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from flask import Flask, render_template_string, jsonify
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(ROOT, "logs", "paper")
TRADES_OPEN = os.path.join(LOGS_DIR, "trades_open.csv")
TRADES_CLOSED = os.path.join(LOGS_DIR, "trades_closed.csv")
REJECTS = os.path.join(LOGS_DIR, "rejects.csv")
ENGINE_START_TXT = os.path.join(LOGS_DIR, "engine_start.txt")

# ---------- helpers ----------
def utcnow():
    return datetime.now(timezone.utc)

def read_engine_start() -> Optional[datetime]:
    try:
        with open(ENGINE_START_TXT, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            return datetime.fromisoformat(line)
    except Exception:
        return None

def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df

def to_utc(df: pd.DataFrame, col: str):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            pass
    return df

def summarize_counts(last_minutes: int = 10) -> Dict:
    now = utcnow()
    open_df = read_csv(TRADES_OPEN)
    close_df = read_csv(TRADES_CLOSED)
    rej_df = read_csv(REJECTS)

    to_utc(open_df, "opened_at"); to_utc(open_df, "expire_at")
    to_utc(close_df, "opened_at"); to_utc(close_df, "closed_at")
    to_utc(rej_df, "ts")

    # totals
    signals_total = len(open_df)  # 열린 포지션 수 = “allowed 신호”로 간주
    rejects_total = len(rej_df)

    # recent window
    cut = now - timedelta(minutes=last_minutes)
    signals_recent = int((open_df["opened_at"] >= cut).sum()) if "opened_at" in open_df.columns else 0
    rejects_recent = int((rej_df["ts"] >= cut).sum()) if "ts" in rej_df.columns else 0

    # open positions (approx): opened rows with same (opened_at,symbol) not in closed
    open_now = 0
    if not open_df.empty:
        key_open = open_df[["opened_at","symbol"]].astype(str).agg("|".join, axis=1)
        key_closed = pd.Series([], dtype=str)
        if not close_df.empty:
            key_closed = close_df[["opened_at","symbol"]].astype(str).agg("|".join, axis=1)
        open_now = int((~key_open.isin(set(key_closed))).sum())

    # engine timestamps
    eng_start = read_engine_start()
    server_uptime = None
    if eng_start:
        server_uptime = now - eng_start

    return {
        "signals_total": int(signals_total),
        "signals_recent": int(signals_recent),
        "rejects_total": int(rejects_total),
        "rejects_recent": int(rejects_recent),
        "open_positions": int(open_now),
        "engine_start": eng_start.isoformat() if eng_start else "",
        "uptime_secs": int(server_uptime.total_seconds()) if server_uptime else None,
        "now": now.isoformat(),
    }

def per_minute_counts(df: pd.DataFrame, time_col: str, last_minutes: int = 60) -> Tuple[List[str], List[int]]:
    if df.empty or time_col not in df.columns:
        # return empty 60-min series
        labels = []
        values = []
        for i in range(last_minutes, 0, -1):
            t = utcnow() - timedelta(minutes=i)
            labels.append(t.strftime("%H:%M"))
            values.append(0)
        return labels, values

    df = df.copy()
    to_utc(df, time_col)
    cut = utcnow() - timedelta(minutes=last_minutes)
    df = df[df[time_col] >= cut]
    if df.empty:
        return per_minute_counts(pd.DataFrame(), time_col, last_minutes)

    df["bucket"] = df[time_col].dt.floor("1min")
    s = df.groupby("bucket").size()

    # build complete series
    labels, values = [], []
    for i in range(last_minutes, 0, -1):
        t = utcnow() - timedelta(minutes=i)
        lbl = t.strftime("%H:%M")
        labels.append(lbl)
        values.append(int(s.get(t.replace(second=0, microsecond=0), 0)))
    return labels, values

def recent_allowed_preview(n: int = 10) -> List[Dict]:
    df = read_csv(TRADES_OPEN)
    to_utc(df, "opened_at")
    if df.empty:
        return []
    df = df.sort_values("opened_at", ascending=False).head(n)
    cols = [c for c in ["opened_at","symbol","event","side","level"] if c in df.columns]
    out = []
    for _, r in df[cols].iterrows():
        out.append({
            "opened_at": (r["opened_at"].isoformat() if isinstance(r["opened_at"], pd.Timestamp) else str(r["opened_at"])),
            "symbol": r.get("symbol",""),
            "event": r.get("event",""),
            "side": r.get("side",""),
            "level": int(r.get("level", 0)) if pd.notna(r.get("level", None)) else "",
        })
    return out

# ---------- web app ----------
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="utf-8">
  <title>Paper Trader Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap & Chart.js via CDN -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { background-color:#0f1115; }
    .card { background:#161a22; border-color:#222634; }
    .mini { height:220px; }
    .muted { color:#94a3b8; }
    .metric { font-size: 1.15rem; }
    .title { font-weight:700; }
    .badge-src { text-transform:uppercase; letter-spacing:.02em; }
  </style>
</head>
<body>
<div class="container-fluid py-3">
  <div class="d-flex align-items-center justify-content-between">
    <h3 class="title">Paper Trader Dashboard</h3>
    <div>
      <div class="btn-group btn-group-sm" role="group" aria-label="refresh">
        <button class="btn btn-outline-secondary" onclick="setRefresh(5)">5s</button>
        <button class="btn btn-outline-secondary" onclick="setRefresh(10)">10s</button>
        <button class="btn btn-outline-secondary" onclick="setRefresh(30)">30s</button>
        <button class="btn btn-outline-secondary" onclick="setRefresh(60)">60s</button>
        <button class="btn btn-outline-warning" onclick="setRefresh(0)">pause</button>
      </div>
    </div>
  </div>

  <div class="row g-3 mt-1">
    <div class="col-lg-4">
      <div class="card p-3">
        <div class="title mb-2">Server & Engine</div>
        <div class="metric"><span class="muted">Server time</span> <span id="now">-</span></div>
        <div class="metric"><span class="muted">Engine start</span> <span id="engine_start">-</span></div>
        <div class="metric"><span class="muted">Engine uptime</span> <span id="uptime">-</span></div>
        <small class="muted">* start시간은 <code>engine_start.txt</code> 기준</small>
      </div>
    </div>
    <div class="col-lg-4">
      <div class="card p-3">
        <div class="title mb-2">Signals</div>
        <div class="metric"><span class="muted">Signals (total)</span> <span id="signals_total">-</span></div>
        <div class="metric"><span class="muted">Signals (recent 10m)</span> <span id="signals_recent">-</span></div>
      </div>
    </div>
    <div class="col-lg-4">
      <div class="card p-3">
        <div class="title mb-2">Rejects & Open positions</div>
        <div class="metric"><span class="muted">Rejects (total)</span> <span id="rejects_total">-</span></div>
        <div class="metric"><span class="muted">Rejects (recent 10m)</span> <span id="rejects_recent">-</span></div>
        <div class="metric"><span class="muted">Open positions</span> <span id="open_positions">-</span></div>
      </div>
    </div>
  </div>

  <div class="row g-3 mt-1">
    <div class="col-lg-6">
      <div class="card p-3">
        <div class="title mb-2">Allowed Signals per minute (last 60m)</div>
        <canvas id="chartAllowed" class="mini"></canvas>
      </div>
    </div>
    <div class="col-lg-6">
      <div class="card p-3">
        <div class="title mb-2">Run-phase Rejects per minute (last 60m)</div>
        <canvas id="chartRejects" class="mini"></canvas>
      </div>
    </div>
  </div>

  <div class="card p-3 mt-3">
    <div class="title mb-2">Recent Allowed Signals (preview)</div>
    <div class="table-responsive">
      <table class="table table-sm table-dark align-middle">
        <thead><tr>
          <th>opened_at</th><th>symbol</th><th>event</th><th>side</th><th>level</th>
        </tr></thead>
        <tbody id="previewBody"></tbody>
      </table>
    </div>
  </div>

  <div class="mt-3 muted">Data sources: <code>logs/paper/trades_open.csv</code>, <code>trades_closed.csv</code>, <code>rejects.csv</code></div>
</div>

<script>
let refreshMs = 10000; // default 10s
let timerId = null;

function setRefresh(sec){
  if (timerId) clearInterval(timerId);
  refreshMs = (sec > 0) ? sec*1000 : 0;
  if (refreshMs > 0) {
    timerId = setInterval(loadAll, refreshMs);
  }
}

async function loadAll(){
  try{
    const s = await fetch('/api/summary').then(r=>r.json());
    document.getElementById('signals_total').innerText = s.signals_total;
    document.getElementById('signals_recent').innerText = s.signals_recent;
    document.getElementById('rejects_total').innerText = s.rejects_total;
    document.getElementById('rejects_recent').innerText = s.rejects_recent;
    document.getElementById('open_positions').innerText = s.open_positions;
    document.getElementById('engine_start').innerText = s.engine_start || '-';
    document.getElementById('now').innerText = s.now || '-';
    document.getElementById('uptime').innerText = (s.uptime_secs!=null)? fmtDur(s.uptime_secs) : '-';

    const a = await fetch('/api/series/allowed').then(r=>r.json());
    const r = await fetch('/api/series/rejects').then(r=>r.json());
    drawChart('chartAllowed', a.labels, a.values, 'rgba(77,171,247,0.9)');
    drawChart('chartRejects', r.labels, r.values, 'rgba(255,146,43,0.9)');

    const p = await fetch('/api/recent_allowed').then(r=>r.json());
    renderPreview(p.rows || []);
  }catch(e){ console.error(e); }
}

function fmtDur(secs){
  const h = Math.floor(secs/3600);
  const m = Math.floor((secs%3600)/60);
  const s = Math.floor(secs%60);
  return `${h}h ${m}m ${s}s`;
}

let charts = {};
function drawChart(id, labels, values, color){
  const ctx = document.getElementById(id).getContext('2d');
  if (charts[id]) { charts[id].destroy(); }
  charts[id] = new Chart(ctx, {
    type: 'line',
    data: { labels: labels, datasets: [{ data: values, borderColor: color, backgroundColor: color, fill:false, tension:0.25, pointRadius:0 }] },
    options: {
      responsive:true,
      plugins:{ legend:{display:false} },
      scales:{
        x:{ ticks:{ color:'#94a3b8' } },
        y:{ ticks:{ color:'#94a3b8' }, beginAtZero:true, suggestedMax: Math.max(1, Math.max(...values)) }
      }
    }
  });
}

function renderPreview(rows){
  const tb = document.getElementById('previewBody');
  tb.innerHTML = '';
  for(const r of rows){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${r.opened_at||''}</td><td>${r.symbol||''}</td><td>${r.event||''}</td><td>${r.side||''}</td><td>${r.level||''}</td>`;
    tb.appendChild(tr);
  }
}

document.addEventListener('DOMContentLoaded', ()=>{
  loadAll();
  setRefresh(10);
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE)

@app.route("/api/summary")
def api_summary():
    return jsonify(summarize_counts(10))

@app.route("/api/series/allowed")
def api_series_allowed():
    df = read_csv(TRADES_OPEN)
    to_utc(df, "opened_at")
    labels, values = per_minute_counts(df, "opened_at", 60)
    return jsonify({"labels": labels, "values": values})

@app.route("/api/series/rejects")
def api_series_rejects():
    df = read_csv(REJECTS)
    to_utc(df, "ts")
    labels, values = per_minute_counts(df, "ts", 60)
    return jsonify({"labels": labels, "values": values})

@app.route("/api/recent_allowed")
def api_recent_allowed():
    return jsonify({"rows": recent_allowed_preview(10)})

if __name__ == "__main__":
    # Flask dev server
    app.run(host="0.0.0.0", port=8091, debug=False)