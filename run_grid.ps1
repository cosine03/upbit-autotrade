# run_grid.ps1
param(
  [string]$Signals = ".\logs\signals_tv_snapshot_YYYYMMDDTHHMMSSZ.csv",  # 통합 스냅샷
  [string]$Prices  = ".\logs\prices.csv",
  [string]$Universe = ".\configs\universe.txt",
  [int]$WindowSec = 600,
  [double]$Fee = 0.001,
  [string]$OutDir = ".\logs\backtest\grid_all"
)

# 이벤트 세트
$EVENT_SETS = @(
  @{ key = "line";     events = "line_breakout" }
  @{ key = "box";      events = "box_breakout" }
  @{ key = "pib";      events = "price_in_box" }
  @{ key = "line_box"; events = "line_breakout,box_breakout" }
  @{ key = "box_pib";  events = "box_breakout,price_in_box" }
  @{ key = "all";      events = "line_breakout,box_breakout,price_in_box" }
)

# 만기(분)
$EXPIRIES = @(5,10,15,30,60,120)

# 사이드
$SIDES = @("support","resistance")

# 출력 디렉토리
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host ("Grid: {0} event-sets x {1} expiries x {2} sides -> out={3}" -f `
            $EVENT_SETS.Count, $EXPIRIES.Count, $SIDES.Count, $OutDir)
Write-Host ("Args: window={0} sec, fee={1}" -f $WindowSec, $Fee)
Write-Host ""

foreach ($side in $SIDES) {
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