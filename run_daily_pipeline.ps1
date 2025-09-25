Param(
  [ValidateSet("AM","PM")][string]$TagHalf = "AM"
)

# ===== Safety =====
$ErrorActionPreference = "Stop"

# ===== Date/Paths =====
$DATE = Get-Date -Format "yyyy-MM-dd"
$TAG  = "${DATE}_$TagHalf"
$OUT  = ".\logs\daily\$TAG"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
New-Item -ItemType Directory -Force -Path ".\logs\signals" | Out-Null
New-Item -ItemType Directory -Force -Path ".\logs\history" | Out-Null

# ===== Load config =====
$cfgPath = ".\pipeline_config.json"
if (-not (Test-Path $cfgPath)) { throw "pipeline_config.json not found." }
$cfg = Get-Content $cfgPath | ConvertFrom-Json

# ===== 0) OHLCV 업데이트 =====
# 심볼 목록은 universe.txt(한 줄에 하나)에서 읽음
$symbolsFile = ".\configs\universe.txt"
if (-not (Test-Path $symbolsFile)) {
  throw "configs\universe.txt not found. Create it with one symbol per line (e.g., KRW-BTC)."
}

Write-Host "[OHLCV] Fetching ..."
python -u .\fetch_ohlcv_upbit.py `
  --symbols-file "$symbolsFile" `
  --timeframe $cfg.timeframe `
  --outdir .\data\ohlcv

# ===== 1) TV enriched 신호 백업 =====
if (-not (Test-Path $cfg.paths.signals_current)) {
  throw "signals_current not found: $($cfg.paths.signals_current)"
}
$archivedSignals = ".\logs\signals\signals_${TAG}.csv"
Copy-Item $cfg.paths.signals_current $archivedSignals -Force
Write-Host "[SIGNALS] Archived -> $archivedSignals"

# ===== 2) (옵션) dist_max/TP,SL 추천 =====
python -u .\dynamic_params.py `
  --signals $archivedSignals `
  --grid-csv $cfg.paths.grid_summary `
  --target-lo 20 --target-hi 30 `
  --events box_breakout,line_breakout,price_in_box `
  --min-trades 30 --metric avg_net `
  --clamp-min 0.0 --clamp-max 0.001 `
  --out "$OUT\dynamic_params.json"

# ===== 3) Breakout-only 백테스트 (dist 고정) =====
python -u $cfg.paths.backtest_script $archivedSignals `
  --timeframe $cfg.timeframe `
  --expiries $cfg.expiries `
  --tp $cfg.tp --sl $cfg.sl --fee $cfg.fee `
  --dist-max $cfg.dist_max `
  --procs 24 `
  --ohlcv-roots $cfg.ohlcv_roots `
  --ohlcv-patterns $cfg.ohlcv_patterns `
  --assume-ohlcv-tz $cfg.assume_ohlcv_tz `
  --outdir "$OUT\bt_breakout_only"

# ===== 4) Box-in Line Breakout (필터 → 백테스트) =====
$filtered = "$OUT\signals_boxin_linebreak.csv"
python -u .\filter_boxin_linebreak.py $archivedSignals `
  --out $filtered `
  --lookback-hours $cfg.lookback_hours `
  --dist-max $cfg.dist_max

python -u $cfg.paths.backtest_script $filtered `
  --timeframe $cfg.timeframe `
  --expiries $cfg.expiries `
  --tp $cfg.tp --sl $cfg.sl --fee $cfg.fee `
  --dist-max 9 `
  --procs 24 `
  --ohlcv-roots $cfg.ohlcv_roots `
  --ohlcv-patterns $cfg.ohlcv_patterns `
  --assume-ohlcv-tz $cfg.assume_ohlcv_tz `
  --outdir "$OUT\bt_boxin_linebreak"

# ===== 5) 일일 리포트(엑셀+차트) =====
python -u .\make_daily_report.py `
  --in1 "$OUT\bt_breakout_only\bt_tv_events_stats_summary.csv" `
  --in2 "$OUT\bt_boxin_linebreak\bt_tv_events_stats_summary.csv" `
  --out "$OUT\daily_report.xlsx" `
  --tag $TAG

# ===== 6) 히스토리 누적 =====
python -u .\append_history.py --summary "$OUT\bt_breakout_only\bt_tv_events_stats_summary.csv" `
  --strategy breakout_only --date $DATE `
  --history ".\logs\history\bt_breakout_only.csv"

python -u .\append_history.py --summary "$OUT\bt_boxin_linebreak\bt_tv_events_stats_summary.csv" `
  --strategy boxin_linebreak --date $DATE `
  --history ".\logs\history\bt_boxin_linebreak.csv"

Write-Host "[OK] Daily pipeline complete -> $OUT"