Param(
  [ValidateSet("AM","PM")][string]$TagHalf = "AM"
)

# ===== Safety =====
$ErrorActionPreference = "Stop"
$env:PANDAS_IGNORE_PYARROW = "1"   # avoid heavy pyarrow import

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
$symbolsFile = ".\configs\universe.txt"
if (-not (Test-Path $symbolsFile)) { throw "configs\universe.txt not found. Create it with one symbol per line (e.g., KRW-BTC)." }

Write-Host "[OHLCV] Fetching ..."
python -u .\fetch_ohlcv_upbit.py `
  --symbols-file "$symbolsFile" `
  --timeframe $cfg.timeframe `
  --outdir .\data\ohlcv

# ===== 1) TV enriched 신호 백업 =====
if (-not (Test-Path $cfg.paths.signals_current)) { throw "signals_current not found: $($cfg.paths.signals_current)" }
$archivedSignals = ".\logs\signals\signals_${TAG}.csv"
Copy-Item $cfg.paths.signals_current $archivedSignals -Force
Write-Host "[SIGNALS] Archived -> $archivedSignals"

# ===== 2) 동적 파라미터 산출 & dist_max 자동 적용 =====
$distToUse = [double]$cfg.dist_max   # default
$dynJson = Join-Path $OUT "dynamic_params.json"
try {
  python -u .\dynamic_params.py `
    --signals $archivedSignals `
    --grid-csv $cfg.paths.grid_summary `
    --target-lo 20 --target-hi 30 `
    --events box_breakout,line_breakout,price_in_box `
    --min-trades 30 --metric avg_net `
    --clamp-min 0.0 --clamp-max 0.001 `
    --out $dynJson

  if (Test-Path $dynJson) {
    $dyn = Get-Content $dynJson | ConvertFrom-Json
    $v = $null
    if ($dyn.dist_max.value_ratio -ne $null) { $v = [double]$dyn.dist_max.value_ratio }
    elseif ($dyn.dist_max.value -ne $null)   { $v = [double]$dyn.dist_max.value }
    if ($v -ne $null) {
      # clamp
      if ($dyn.dist_max.clamp.min -ne $null -and $v -lt [double]$dyn.dist_max.clamp.min) { $v = [double]$dyn.dist_max.clamp.min }
      if ($dyn.dist_max.clamp.max -ne $null -and $v -gt [double]$dyn.dist_max.clamp.max) { $v = [double]$dyn.dist_max.clamp.max }
      $distToUse = $v
    }
  }
} catch {
  Write-Warning "[DYNAMIC] dynamic estimation failed. Using default dist_max=$distToUse"
}
Write-Host "[DYNAMIC] dist_max to use =" $distToUse

# ===== 3) Breakout-only 사전필터 (box_breakout + line_breakout) =====
$boCsv = "$OUT\signals_breakout_only.csv"

# CSV 읽어서 event가 box_breakout/line_breakout인 것만 필터
Import-Csv -Path $archivedSignals |
  Where-Object { $_.event -in @("box_breakout","line_breakout") } |
  Export-Csv -Path $boCsv -NoTypeInformation -Encoding UTF8

# 개수 로그
$boCount = (Import-Csv -Path $boCsv | Measure-Object).Count
Write-Host "[PRE] breakout-only rows:" $boCount

# ===== 4) Breakout-only 백테스트 (dist = dynamic) =====
python -u $cfg.paths.backtest_script $boCsv `
  --timeframe $cfg.timeframe `
  --expiries $cfg.expiries `
  --tp $cfg.tp --sl $cfg.sl --fee $cfg.fee `
  --dist-max $distToUse `
  --procs 24 `
  --ohlcv-roots $cfg.ohlcv_roots `
  --ohlcv-patterns $cfg.ohlcv_patterns `
  --assume-ohlcv-tz $cfg.assume_ohlcv_tz `
  --outdir "$OUT\bt_breakout_only"

# ===== 5) Box-in Line Breakout (필터 → 백테스트) =====
$filtered = "$OUT\signals_boxin_linebreak.csv"
python -u .\filter_boxin_linebreak.py $archivedSignals `
  --out $filtered `
  --lookback-hours $cfg.lookback_hours `
  --dist-max $distToUse

if (Test-Path $filtered) {
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
} else {
  Write-Warning "[SKIP] filtered CSV not found: $filtered"
}

# ===== 6) 리포트(엑셀+차트+Equity/MDD) =====
$sum1 = "$OUT\bt_breakout_only\bt_tv_events_stats_summary.csv"
$sum2 = "$OUT\bt_boxin_linebreak\bt_tv_events_stats_summary.csv"
$tr1  = "$OUT\bt_breakout_only\bt_tv_events_trades.csv"
$tr2  = "$OUT\bt_boxin_linebreak\bt_tv_events_trades.csv"
if ((Test-Path $sum1) -and (Test-Path $sum2) -and (Test-Path $tr1) -and (Test-Path $tr2)) {
  $reportXlsx = "$OUT\daily_report.xlsx"
  $env:MPLBACKEND="Agg"
  python -u .\make_daily_report.py `
    --in1 $sum1 --in2 $sum2 `
    --trades1 $tr1 --trades2 $tr2 `
    --out $reportXlsx `
    --tag $TAG
} else {
  Write-Warning "[SKIP] report: summary/trades files missing"
}

# ===== 7) 히스토리 누적 =====
if (Test-Path $sum1) {
  python -u .\append_history.py --summary $sum1 `
    --strategy breakout_only --date $DATE `
    --history ".\logs\history\bt_breakout_only.csv"
} else { Write-Warning "[SKIP] history breakout_only: $sum1 missing" }

if (Test-Path $sum2) {
  python -u .\append_history.py --summary $sum2 `
    --strategy boxin_linebreak --date $DATE `
    --history ".\logs\history\bt_boxin_linebreak.csv"
} else { Write-Warning "[SKIP] history boxin_linebreak: $sum2 missing" }

Write-Host "[OK] Daily pipeline complete -> $OUT"