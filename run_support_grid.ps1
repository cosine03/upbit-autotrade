# ===== 사용자 설정 =====
$PYTHON      = "python"                                   # venv 활성화된 셸에서 실행 권장
$ROOT        = "."                                        # repo 루트
$SCRIPT      = Join-Path $ROOT "replay_backtest.py"
$SIGNALS_CSV = Join-Path $ROOT "logs\signals_tv.csv"
$PRICES_CSV  = Join-Path $ROOT "logs\prices.csv"
$UNIVERSE_TXT= Join-Path $ROOT "configs\universe.txt"

$OUT_DIR     = Join-Path $ROOT "logs\backtest\grid_support"
$LOG_PATH    = Join-Path $OUT_DIR "run.log"
$SUMMARY_CSV = Join-Path $OUT_DIR "summary.csv"

# 공통 파라미터
$FEE         = 0.001
$PRICE_WIN   = 600          # REST 가격 윈도(초)
$SIDES       = "support"    # 이번 그리드는 support만

# 만기 후보 (분)
$EXPIRIES = @(5, 10, 15, 30, 60, 120)

# 이벤트 세트(모든 경우의 수)
# key: 요약 이름 / events: 실제 인자 값
$EVENT_SETS = @(
  @{ key = "line";          events = "line_breakout" }
  @{ key = "box";           events = "box_breakout" }
  @{ key = "pib";           events = "price_in_box" }
  @{ key = "line_box";      events = "line_breakout,box_breakout" }
  @{ key = "box_pib";       events = "box_breakout,price_in_box" }
  @{ key = "all";           events = "line_breakout,box_breakout,price_in_box" }
)

# ===== 준비 =====
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
"=== GRID RUN @ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $LOG_PATH -Append

# 요약 CSV 헤더 (없으면 생성)
if (-not (Test-Path $SUMMARY_CSV)) {
  "ts,side,expiry_min,events_key,events,trades,win_pct,avg_pnl,out_csv" | Set-Content -Path $SUMMARY_CSV -Encoding UTF8
}

# ===== 실행 루프 =====
foreach ($exp in $EXPIRIES) {
  foreach ($es in $EVENT_SETS) {
    $key    = $es.key
    $events = $es.events

    $tag    = "{0}_exp{1}" -f $key, $exp
    $outCsv = Join-Path $OUT_DIR ("support_{0}.csv" -f $tag)

    $args = @(
      $SCRIPT,
      "--signals-csv", $SIGNALS_CSV,
      "--prices-csv",  $PRICES_CSV,
      "--universe",    $UNIVERSE_TXT,
      "--sides",       $SIDES,
      "--events",      $events,
      "--expiry-min",  $exp,
      "--price-window-sec", $PRICE_WIN,
      "--fee",         $FEE,
      "--out",         $outCsv
    )

    ">>> RUN  side=$SIDES  exp=${exp}m  events=$events  -> $outCsv" | Tee-Object -FilePath $LOG_PATH -Append

    # 실행 + 표준출력 캡처
    $proc = & $PYTHON @args 2>&1
    $proc | Tee-Object -FilePath $LOG_PATH -Append | Out-Host

    # 성과 파싱 (stdout에서 "Trades:" 라인)
    # 예: "Trades: 55 | Win%: 49.1% | Avg PnL: 0.23%"
    $line = $proc | Select-String -Pattern "^\s*Trades:\s*\d+\s*\|\s*Win%:\s*[\d\.]+%\s*\|\s*Avg PnL:\s*-?[\d\.]+%" -AllMatches | Select-Object -Last 1
    if ($line) {
      $m = [regex]::Match($line.ToString(), "Trades:\s*(\d+)\s*\|\s*Win%:\s*([\d\.]+)%\s*\|\s*Avg PnL:\s*(-?[\d\.]+)%")
      if ($m.Success) {
        $trades = $m.Groups[1].Value
        $winPct = $m.Groups[2].Value
        $avgPnl = $m.Groups[3].Value
        $ts     = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")
        "$ts,$SIDES,$exp,$key,""$events"",$trades,$winPct,$avgPnl,""$outCsv""" | Add-Content -Path $SUMMARY_CSV -Encoding UTF8
      } else {
        # 파싱 실패시 로그
        "WARN: summary parse failed for $tag" | Tee-Object -FilePath $LOG_PATH -Append
      }
    } else {
      "WARN: no 'Trades:' line for $tag" | Tee-Object -FilePath $LOG_PATH -Append
    }

    # 너무 빠른 연속 호출 방지 (API/IO 여유)
    Start-Sleep -Seconds 1
  }
}

"=== DONE @ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $LOG_PATH -Append

# 빠른 요약 미리보기 (선택)
"`nTOP GRID SUMMARY (last 12 rows):" | Out-Host
Import-Csv $SUMMARY_CSV | Select-Object -Last 12 | Format-Table ts, side, expiry_min, events_key, trades, win_pct, avg_pnl