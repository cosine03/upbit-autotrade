Param(
  [ValidateSet("AM","PM")][string]$TagHalf = "AM"
)

$DATE = Get-Date -Format "yyyy-MM-dd"
$TAG  = "${DATE}_$TagHalf"
$OUT  = ".\logs\daily\$TAG"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$cfg = Get-Content ".\pipeline_config.json" | ConvertFrom-Json

# 1) Backup signals
Copy-Item $cfg.paths.signals_current ".\logs\signals\signals_${TAG}.csv"

# 2) Dynamic params (reference-only grid)
python -u .\dynamic_params.py `
  --signals ".\logs\signals\signals_${TAG}.csv" `
  --grid-csv $cfg.paths.grid_summary `
  --target-lo 20 --target-hi 30 `
  --events box_breakout,line_breakout,price_in_box `
  --min-trades 30 --metric avg_net `
  --clamp-min 0.0 --clamp-max 0.001 `
  --out "$OUT\dynamic_params.json"

# 3) Breakout-only
python -u $cfg.paths.backtest_script ".\logs\signals\signals_${TAG}.csv" `
  --timeframe $cfg.timeframe `
  --expiries $cfg.expiries `
  --tp $cfg.tp --sl $cfg.sl --fee $cfg.fee `
  --dist-max $cfg.dist_max `
  --procs 24 `
  --ohlcv-roots $cfg.ohlcv_roots `
  --ohlcv-patterns $cfg.ohlcv_patterns `
  --assume-ohlcv-tz $cfg.assume_ohlcv_tz `
  --outdir "$OUT\bt_breakout_only"

# 4) Box-in Line Breakout
python -u .\filter_boxin_linebreak.py ".\logs\signals\signals_${TAG}.csv" `
  --out "$OUT\signals_boxin_linebreak.csv" `
  --lookback-hours $cfg.lookback_hours `
  --dist-max $cfg.dist_max

python -u $cfg.paths.backtest_script "$OUT\signals_boxin_linebreak.csv" `
  --timeframe $cfg.timeframe `
  --expiries $cfg.expiries `
  --tp $cfg.tp --sl $cfg.sl --fee $cfg.fee `
  --dist-max 9 `
  --procs 24 `
  --ohlcv-roots $cfg.ohlcv_roots `
  --ohlcv-patterns $cfg.ohlcv_patterns `
  --assume-ohlcv-tz $cfg.assume_ohlcv_tz `
  --outdir "$OUT\bt_boxin_linebreak"

# 5) Report
python -u .\make_daily_report.py `
  --in1 "$OUT\bt_breakout_only\bt_tv_events_stats_summary.csv" `
  --in2 "$OUT\bt_boxin_linebreak\bt_tv_events_stats_summary.csv" `
  --out "$OUT\daily_report.xlsx" `
  --tag $TAG

# 6) Append history
python -u .\append_history.py --summary "$OUT\bt_breakout_only\bt_tv_events_stats_summary.csv" `
  --strategy breakout_only --date $DATE `
  --history ".\logs\history\bt_breakout_only.csv"

python -u .\append_history.py --summary "$OUT\bt_boxin_linebreak\bt_tv_events_stats_summary.csv" `
  --strategy boxin_linebreak --date $DATE `
  --history ".\logs\history\bt_boxin_linebreak.csv"

Write-Host "[OK] Daily pipeline complete -> $OUT"