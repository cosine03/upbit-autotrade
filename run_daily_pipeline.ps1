Param(
  [ValidateSet("AM","PM")][string]$TagHalf = "AM"
)

# venv Python 강제 사용 (스크립트 범위 alias)
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { Set-Alias -Name python -Value $venvPy -Scope Script }

# 공통 경로
$ROOT = $PSScriptRoot
$OUT  = Join-Path $ROOT "logs\daily\$((Get-Date).ToString('yyyy-MM-dd'))_$TagHalf"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$DATE = (Get-Date).ToString('yyyy-MM-dd')
$TAG  = "${DATE}_$TagHalf"

Write-Host "[INFO] ==== PIPELINE START ($TAG) ===="

# ===== 1) OHLCV (Upbit) =====
#   ※ 필요시 기존 그대로 두세요
$cfg = @{
  timeframe = "15m"
  paths = @{
    backtest_script = ".\bt_tv_events.py"
    grid_summary    = ".\logs\history\grid_summary.csv"
  }
}

python -u .\fetch_ohlcv_upbit.py `
  --timeframe $cfg.timeframe `
  --out-dir ".\logs\ohlcv\upbit" `
  --limit 500

# ===== 2) dynamic_params (dist_max 산출) =====
$dynJson = Join-Path $OUT "dynamic_params.json"
$distToUse = $null
try {
  python -u .\dynamic_params.py `
    --grid-csv $cfg.paths.grid_summary `
    --out-json $dynJson `
    --clamp-min 0.0 --clamp-max 0.001

  if (Test-Path $dynJson) {
    $dyn = Get-Content $dynJson | ConvertFrom-Json
    $distToUse = [double]$dyn.dist_max.value

    # clamp
    if ($dyn.dist_max.clamp.min -ne $null -and $distToUse -lt [double]$dyn.dist_max.clamp.min) { $distToUse = [double]$dyn.dist_max.clamp.min }
    if ($dyn.dist_max.clamp.max -ne $null -and $distToUse -gt [double]$dyn.dist_max.clamp.max) { $distToUse = [double]$dyn.dist_max.clamp.max }
  }
} catch { Write-Warning "[DYNAMIC] dynamic estimation failed. $_" }

if (-not $distToUse) { $distToUse = 0.00025 }   # 백테스트 기본값 (보수적)
Write-Host "[DYNAMIC] dist_max to use =" $distToUse

# ===== 3) (신규) TV 로그에서 이번 창(AM/PM 12h) 시그널 생성 =====
#   - 입력: .\logs\signals_tv.csv  (전체 누적)
#   - 출력: .\logs\signals\signals_${DATE}_${TagHalf}.csv (해당 12시간만)
$tvAllCsv = ".\logs\signals_tv.csv"
$archSignalsDir = ".\logs\signals"
New-Item -ItemType Directory -Force -Path $archSignalsDir | Out-Null
$archivedSignals = Join-Path $archSignalsDir "signals_${TAG}.csv"

function Get-WindowBoundariesUtc([string]$tagHalf) {
  # 로컬 08:00 또는 20:00를 기준으로 최근 12시간 창을 만들어 UTC로 변환
  $fmt = 'yyyy-MM-dd HH:mm'
  if ($tagHalf -eq 'AM') {
    $endLocal   = [datetime]::ParseExact("$DATE 08:00",$fmt,$null)
  } else {
    $endLocal   = [datetime]::ParseExact("$DATE 20:00",$fmt,$null)
  }
  $startLocal = $endLocal.AddHours(-12)
  $startUtc = [datetimeoffset]::new($startLocal).ToUniversalTime()
  $endUtc   = [datetimeoffset]::new($endLocal).ToUniversalTime()
  return ,@($startUtc, $endUtc)
}

$win = Get-WindowBoundariesUtc $TagHalf
$winStartUtc = $win[0]
$winEndUtc   = $win[1]
Write-Host ("[SIGNALS] window UTC = {0:s} ~ {1:s}" -f $winStartUtc.UtcDateTime, $winEndUtc.UtcDateTime)

if (Test-Path $tvAllCsv) {
  $raw = Import-Csv $tvAllCsv
  # ISO8601(+00:00) 파싱 → UTC 구간 필터
  $filtered = $raw | Where-Object {
    try {
      $ts = [datetimeoffset]::Parse($_.ts)
      ($ts -ge $winStartUtc) -and ($ts -lt $winEndUtc)
    } catch { $false }
  }

  # 참고용: price_in_box 포함 / 본 매매는 resistance + breakout 위주
  # 여기서는 그대로 저장하고, 백테스트 단계에서 이벤트 필터를 나눠서 처리
  $filtered | Export-Csv -Path $archivedSignals -NoTypeInformation -Encoding UTF8

  $h = (Get-FileHash $archivedSignals).Hash
  $n = ($filtered | Measure-Object).Count
  Write-Host "[SIGNALS] saved $n rows -> $archivedSignals"
  Write-Host "[SIGNALS] hash=" $h
} else {
  Write-Warning "[SIGNALS] TV log not found: $tvAllCsv (skip)"
}

# ===== 4) Breakout-only 백테스트 (dist = dynamic) =====
$boCsv = "$OUT\signals_breakout_only.csv"
if (Test-Path $archivedSignals) {
  Import-Csv -Path $archivedSignals |
    Where-Object { $_.event -in @('box_breakout','line_breakout') -and $_.side -eq 'resistance' } |
    Export-Csv -Path $boCsv -NoTypeInformation -Encoding UTF8
  $boCount = (Import-Csv -Path $boCsv | Measure-Object).Count
  Write-Host "[BACKTEST] breakout_only input rows =" $boCount
} else {
  Write-Warning "[BACKTEST] archived signals not found; skip breakout_only"
}

if (Test-Path $boCsv) {
  python -u $cfg.paths.backtest_script $boCsv `
    --timeframe $cfg.timeframe `
    --dist-min 0.0 --dist-max $distToUse `
    --expiries "0.5,1,2" `
    --long-only `
    --out-dir "$OUT\bt_breakout_only"
}

# ===== 5) 박스안에서 → 라인브레이크 필터 백테스트 =====
$filteredBL = "$OUT\signals_boxin_linebreak.csv"
if (Test-Path $archivedSignals) {
  python -u .\filter_boxin_linebreak.py $archivedSignals `
    --within-box-ts "1800" `
    --out $filteredBL
  if (Test-Path $filteredBL) {
    $c2 = (Import-Csv $filteredBL | Measure-Object).Count
    Write-Host "[BACKTEST] boxin_linebreak input rows =" $c2
    python -u $cfg.paths.backtest_script $filteredBL `
      --timeframe $cfg.timeframe `
      --dist-min 0.0 --dist-max $distToUse `
      --expiries "0.5,1,2" `
      --long-only `
      --out-dir "$OUT\bt_boxin_linebreak"
  } else {
    Write-Warning "[SKIP] filtered CSV not found: $filteredBL"
  }
}

# ===== 6.5) 리포트 요약용 추가 (optional) =====

# Dynamic Params 요약
$dynJson = Join-Path $OUT "dynamic_params.json"
if (Test-Path $dynJson) {
  try {
    $dyn = Get-Content $dynJson | ConvertFrom-Json
    $suggested = $dyn.dist_max.value
    $applied   = $distToUse
    Write-Host ("[REPORT] dist_max suggested={0}, applied={1}" -f $suggested, $applied)

    Add-Content -Path (Join-Path $OUT "report_extras.txt") `
      -Value ("DYNAMIC DIST: suggested={0}, applied={1}" -f $suggested, $applied)
  } catch {
    Write-Warning "[REPORT] dynamic_params.json parse failed"
  }
}

# Universe 변경 요약
$univHist = ".\logs\history\universe_changes.csv"
if (Test-Path $univHist) {
  $lastLine = Get-Content $univHist | Select-Object -Last 1
  Add-Content -Path (Join-Path $OUT "report_extras.txt") -Value ("UNIVERSE LAST CHANGE: {0}" -f $lastLine)
}

# ===== 6) 리포트/히스토리 업데이트 =====
$sum1 = "$OUT\bt_breakout_only\bt_tv_events_stats_summary.csv"
$sum2 = "$OUT\bt_boxin_linebreak\bt_tv_events_stats_summary.csv"
$tr1  = "$OUT\bt_breakout_only\bt_tv_events_trades.csv"
$tr2  = "$OUT\bt_boxin_linebreak\bt_tv_events_trades.csv"

if (Test-Path $sum1 -or Test-Path $sum2) {
  python -u .\make_daily_report.py `
    --date $DATE --half $TagHalf `
    --summary1 $sum1 --summary2 $sum2 `
    --trades1 $tr1   --trades2 $tr2 `
    --out-dir $OUT
}

if (Test-Path $sum1) {
  python -u .\append_history.py --summary $sum1 `
    --history ".\logs\history\bt_breakout_only.csv"
}
if (Test-Path $sum2) {
  python -u .\append_history.py --summary $sum2 `
    --history ".\logs\history\bt_boxin_linebreak.csv"
}

Write-Host "[INFO] ==== PIPELINE END ($TAG) ===="