param(
  [ValidateSet('AM','PM')]
  [string]$TagHalf = 'AM',
  [string]$Root    = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Ensure-Dir($p) {
  if ([string]::IsNullOrWhiteSpace($p)) { throw "Ensure-Dir got null/empty path" }
  if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

function Must-Exist($label, $p) {
  if ([string]::IsNullOrWhiteSpace($p)) { throw "$label path is null/empty" }
  if (-not (Test-Path -LiteralPath $p)) { throw "$label not found: $p" }
}

$DATE     = Get-Date -Format 'yyyy-MM-dd'
$DailyDir = Join-Path $Root "logs\daily\${DATE}_$TagHalf"

# === 입력/출력 경로 정의 (모두 Join-Path로 명시) ===
$SignalsTV           = Join-Path $Root "logs\signals_tv.csv"
$SignalsBreakout     = Join-Path $DailyDir "signals_breakout_only.csv"
$SignalsBoxLine      = Join-Path $DailyDir "signals_boxin_linebreak.csv"
$DynParamsJson       = Join-Path $DailyDir "dynamic_params.json"

$BtDirBreakout       = Join-Path $DailyDir "bt_breakout_only"
$BtDirBoxLine        = Join-Path $DailyDir "bt_boxin_linebreak"
$MergedSummaryOut    = Join-Path $DailyDir "bt_stats_summary_merged_${TagHalf}.csv"

# Ensure-Dir $DailyDir ...
if (-not (Test-Path -LiteralPath $DynParamsJson)) {
  Write-Host "dynamic_params.json not found. Initializing with defaults..."
  @{
    timeframe = "15m"
    expiries  = @("0.5h","1h","2h")
    tp        = 1.75
    sl        = 0.7
    fee       = 0.001
    entry     = "prev_close"
    dist_max  = 0.00018879720703916502
  } | ConvertTo-Json | Set-Content -Encoding UTF8 $DynParamsJson
}

# === 가드/진단 출력 ===
Write-Host "== RUN START == Root=$Root  Half=$TagHalf  ==="
Write-Host "DATE            : $DATE"
Write-Host "DailyDir        : $DailyDir"
Write-Host "SignalsTV       : $SignalsTV"
Write-Host "SignalsBreakout : $SignalsBreakout"
Write-Host "SignalsBoxLine  : $SignalsBoxLine"
Write-Host "DynParamsJson   : $DynParamsJson"
Write-Host "BtDirBreakout   : $BtDirBreakout"
Write-Host "BtDirBoxLine    : $BtDirBoxLine"
Write-Host "MergedSummaryOut: $MergedSummaryOut"

# 필수 폴더/파일 확인
Ensure-Dir $DailyDir
Ensure-Dir $BtDirBreakout
Ensure-Dir $BtDirBoxLine
Must-Exist "signals_tv.csv" $SignalsTV
Must-Exist "dynamic_params.json" $DynParamsJson
Must-Exist "signals_breakout_only.csv" $SignalsBreakout
Must-Exist "signals_boxin_linebreak.csv" $SignalsBoxLine

# === 백테스트 호출 예시 ===
# 필요에 따라 네가 이미 쓰는 backtest 호출 라인을 그대로 두고, 위 경로 변수만 교체하면 됨.
# python .\backtest_tv_events_mp.py $SignalsBreakout     --outdir $BtDirBreakout     ...
# python .\backtest_tv_events_mp.py $SignalsBoxLine      --outdir $BtDirBoxLine      ...

# === 성능 요약 합치기 (한 줄) ===
# PowerShell 한 줄 버전 (파이썬 없어도 됨). 두 summary 합쳐 저장.
$sum1 = Join-Path $BtDirBreakout "bt_tv_events_stats_summary.csv"
$sum2 = Join-Path $BtDirBoxLine  "bt_tv_events_stats_summary.csv"
Must-Exist "bt summary (breakout_only)" $sum1
Must-Exist "bt summary (boxin_linebreak)" $sum2

$csv1 = Import-Csv -LiteralPath $sum1 | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue 'breakout_only' -PassThru }
$csv2 = Import-Csv -LiteralPath $sum2 | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue 'boxin_linebreak' -PassThru }
$all  = @()
$all += $csv1
$all += $csv2
$all | Export-Csv -LiteralPath $MergedSummaryOut -NoTypeInformation -Encoding UTF8

Write-Host "merged summary saved -> $MergedSummaryOut"
Write-Host "== DONE =="

# == 메일 발송 ==
$Subject = "[Upbit Daily $TagHalf] $DATE"
$Body    = @"
$DATE $TagHalf 리포트입니다.

- breakout_only / boxin_linebreak 요약 CSV를 첨부했습니다.
- 통합 요약: $(Split-Path $MergedSummaryOut -Leaf)
"@

$attachList = @()
$attachList += $MergedSummaryOut
$attachB = Join-Path $BtDirBreakout   "bt_tv_events_stats_summary.csv"
$attachL = Join-Path $BtDirBoxLine    "bt_tv_events_stats_summary.csv"
if (Test-Path $attachB) { $attachList += $attachB }
if (Test-Path $attachL) { $attachList += $attachL }

$attachArgs = @()
foreach ($a in $attachList) { $attachArgs += @("--attach", $a) }

& .\.venv\Scripts\python.exe .\send_email.py --subject $Subject --body $Body @attachArgs