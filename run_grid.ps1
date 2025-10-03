# run_grid.ps1 (교체)
param(
  [string]$Signals = ".\logs\signals_tv_snapshot_YYYYMMDDTHHMMSSZ.csv",
  [string]$Prices  = ".\logs\prices.csv",
  [string]$Universe = ".\configs\universe.txt",
  [int]$WindowSec = 600,
  [double]$Fee = 0.001,
  [string]$OutDir = ".\logs\backtest\grid_all",
  [string[]]$Sides = @("support","resistance")   # ← 기본값 보장
)

# sanity check
if (-not (Test-Path $Signals)) { Write-Host "ERROR: Signals not found: $Signals"; exit 1 }
if (-not (Test-Path $Prices))  { Write-Host "ERROR: Prices not found:  $Prices";  exit 1 }
if (-not (Test-Path $Universe)){ Write-Host "ERROR: Universe not found: $Universe"; exit 1 }
if (-not $Sides -or $Sides.Count -eq 0) { $Sides = @("support","resistance") }  # ← 가드

# 이벤트 세트
$EVENT_SETS = @(
  @{ key = "line";     events = "line_breakout" },
  @{ key = "box";      events = "box_breakout" },
  @{ key = "pib";      events = "price_in_box" },
  @{ key = "line_box"; events = "line_breakout,box_breakout" },
  @{ key = "box_pib";  events = "box_breakout,price_in_box" },
  @{ key = "all";      events = "line_breakout,box_breakout,price_in_box" }
)

# 만기(분)
$EXPIRIES = @(5,10,15,30,60,120)

# 출력 디렉토리
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host ("Grid: {0} event-sets x {1} expiries x {2} sides -> out={3}" -f `
            $EVENT_SETS.Count, $EXPIRIES.Count, $Sides.Count, $OutDir)
Write-Host ("Args: window={0} sec, fee={1}" -f $WindowSec, $Fee)
Write-Host ("Using sides: {0}" -f ($Sides -join ","))  # 디버그용
Write-Host ""

foreach ($side in $Sides) {
  foreach ($es in $EVENT_SETS) {
    $key = $es.key
    $events = $es.events
    foreach ($m in $EXPIRIES) {
      $outFile = Join-Path $OutDir ("{0}_{1}_exp{2}.csv" -f $side, $key, $m)
      $cmd = @(
        ".\replay_backtest.py",
        "--signals-csv", $Signals,
        "--prices-csv",  $Prices,
        "--universe",    $Universe,
        "--sides",       $side,
        "--events",      $events,
        "--expiry-min",  $m,
        "--price-window-sec", $WindowSec,
        "--fee",         $Fee,
        "--out",         $outFile
      )
      Write-Host "==> $($cmd -join ' ')"
      python @cmd
      Write-Host ("done -> {0}" -f $outFile)
      Write-Host ""
    }
  }
}

Write-Host "All done."