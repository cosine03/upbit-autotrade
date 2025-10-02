# run_support_grid.ps1
# Support 시그널용 이벤트 조합 x 만기 조합 그리드 백테스트 러너

param(
  [string]$SignalsCsv = ".\logs\signals_tv.csv",
  [string]$PricesCsv  = ".\logs\prices.csv",
  [string]$Universe   = ".\configs\universe.txt",
  [string]$OutDir     = ".\logs\backtest\support_grid",
  [int[]] $Expiries   = @(5,10,15,30,60,120),
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# 이벤트 세트 (여기 쉼표 필수)
$EVENT_SETS = @(
  @{ key = "line";     events = @("line_breakout") },
  @{ key = "box";      events = @("box_breakout") },
  @{ key = "pib";      events = @("price_in_box") },
  @{ key = "line_box"; events = @("line_breakout","box_breakout") },
  @{ key = "box_pib";  events = @("box_breakout","price_in_box") },
  @{ key = "all";      events = @("line_breakout","box_breakout","price_in_box") }
)

# 출력 디렉토리 준비
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Write-Host "Grid: $($EVENT_SETS.Count) event-sets x $($Expiries.Count) expiries -> out=$OutDir"

foreach ($ev in $EVENT_SETS) {
  $evKey = $ev.key
  $evArg = ($ev.events -join ",")   # replay_backtest.py --events 형식에 맞춤

  foreach ($m in $Expiries) {
    $tag     = "support_${evKey}_exp${m}"
    $outFile = Join-Path $OutDir "$tag.csv"

    $argsList = @(
      ".\replay_backtest.py",
      "--signals-csv", $SignalsCsv,
      "--prices-csv",  $PricesCsv,
      "--universe",    $Universe,
      "--sides",       "support",
      "--events",      $evArg,
      "--expiry-min",  $m,
      "--price-window-sec", 600,
      "--fee",         0.001,
      "--out",         $outFile
    )

    Write-Host ("`n==> {0}" -f ($argsList -join " "))

    if ($DryRun) {
      continue
    }

    # 실행
    & python @argsList

    if ($LASTEXITCODE -ne 0) {
      Write-Warning "run failed for $tag (exit=$LASTEXITCODE)"
    } else {
      Write-Host "done -> $outFile"
    }
  }
}

Write-Host "`nAll done."