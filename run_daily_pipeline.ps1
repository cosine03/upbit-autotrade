Param(
  [ValidateSet("AM","PM")][string]$TagHalf = "AM"
)

# ===== Safety / Env =====
$ErrorActionPreference = "Stop"
# venv Python 강제
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { Set-Alias -Name python -Value $venvPy -Scope Script }
# pandas가 pyarrow 끌어오는 지연 방지
$env:PANDAS_IGNORE_PYARROW = "1"

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
if (-not (Test-Path $symbolsFile)) { throw "configs\universe.txt not found." }

Write-Host "[OHLCV] Fetching ..."
python -u .\fetch_ohlcv_upbit.py `
  --symbols-file "$symbolsFile" `
  --timeframe $cfg.timeframe `
  --outdir .\data\ohlcv
if ($LASTEXITCODE -ne 0) { throw "fetch_ohlcv_upbit failed" }

# ===== 1) TV enriched 신호 백업 =====
if (-not (Test-Path $cfg.paths.signals_current)) { throw "signals_current not found: $($cfg.paths.signals_current)" }
$archivedSignals = ".\logs\signals\signals_${TAG}.csv"
Copy-Item $cfg.paths.signals_current $archivedSignals -Force
Write-Host "[SIGNALS] Archived -> $archivedSignals"

# ===== 2) 동적 dist_max 산출 (자동 적용) =====
$distToUse = [double]$cfg.dist_max   # default fallback
$dynJson = Join-Path $OUT "dynamic_params.json"

python -u .\dynamic_params.py `
  --signals $archivedSignals `
  --grid-csv $cfg.paths.grid_summary `
  --target-lo 20 --target-hi 30 `
  --events box_breakout,line_breakout,price_in_box `
  --min-trades 30 --metric avg_net `
  --clamp-min 0.0 --clamp-max 0.001 `
  --out $dynJson
if ($LASTEXITCODE -ne 0) { Write-Warning "[DYNAMIC] estimation failed. Using default dist_max=$distToUse" }

if (Test-Path $dynJson) {
  try {
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
  } catch {
    Write-Warning "[DYNAMIC] read failed. keep default dist_max=$distToUse"
  }
}
Write-Host "[DYNAMIC] dist_max to use =" $distToUse

# ===== 3) Box-in Line Breakout (필터) =====
$filtered = "$OUT\signals_boxin_linebreak.csv"
python -u .\filter_boxin_linebreak.py $archivedSignals `
  --out $filtered `
  --lookback-hours $cfg.lookback_hours `
  --dist-max $distToUse
if ($LASTEXITCODE -ne 0) { throw "filter_boxin_linebreak failed" }
if (-not (Test-Path $filtered)) { throw "filtered CSV not found: $filtered" }

# ===== 4) 백테스트 (box-in line_breakout 전용) =====
$btOut = "$OUT\bt_boxin_linebreak"
python -u $cfg.paths.backtest_script $filtered `
  --timeframe $cfg.timeframe `
  --expiries $cfg.expiries `
  --tp $cfg.tp --sl $cfg.sl --fee $cfg.fee `
  --dist-max 9 `
  --procs 24 `
  --ohlcv-roots $cfg.ohlcv_roots `
  --ohlcv-patterns $cfg.ohlcv_patterns `
  --assume-ohlcv-tz $cfg.assume_ohlcv_tz `
  --outdir $btOut
if ($LASTEXITCODE -ne 0) { throw "backtest failed" }

# ===== 5) 리포트(단일 전략) =====
$sum = "$btOut\bt_tv_events_stats_summary.csv"
$trd = "$btOut\bt_tv_events_trades.csv"
if (-not (Test-Path $sum)) { throw "summary missing: $sum" }
if (-not (Test-Path $trd)) { throw "trades missing: $trd" }

$reportXlsx = "$OUT\daily_report.xlsx"
$env:MPLBACKEND="Agg"
python -u .\make_daily_report.py `
  --summary $sum `
  --trades  $trd `
  --out $reportXlsx `
  --tag $TAG `
  --strategy boxin_linebreak
if ($LASTEXITCODE -ne 0) { throw "report failed" }

# ===== 6) 히스토리 누적 =====
python -u .\append_history.py --summary $sum `
  --strategy boxin_linebreak --date $DATE `
  --history ".\logs\history\bt_boxin_linebreak.csv"
if ($LASTEXITCODE -ne 0) { throw "append_history failed" }

# ===== 7) 요약 출력 =====
Write-Host "`n[OK] Box-in LineBreak pipeline complete -> $OUT"
try {
  $dyn = Get-Content $dynJson | ConvertFrom-Json
  Write-Host "[OK] dist_max: $($dyn.dist_max.value_ratio) (~$($dyn.dist_max.value_percent)%) / est_count=$($dyn.dist_max.est_count)"
} catch {}
Write-Host "[OK] summary/trades saved -> $sum / $trd"
Write-Host "[OK] report saved -> $reportXlsx"