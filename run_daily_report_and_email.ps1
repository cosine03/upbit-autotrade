<#
.SYNOPSIS
  (AM/PM) 일일 리포트 생성 후 이메일로 자동 발송하는 통합 스크립트

.PARAMETER TagHalf
  AM 또는 PM

.PARAMETER Root
  프로젝트 루트 (기본: D:\upbit_autotrade_starter)

.PARAMETER RunPipeline
  리포트 전에 전체 일일 파이프라인(run_daily_pipeline.ps1)도 실행할지 여부

.PARAMETER Subject
  이메일 제목 접두어

.PARAMETER AttachGlob
  첨부 파일 패턴(세미콜론 구분)

.PARAMETER To
  수신자(미지정 시 .env의 MAIL_TO 사용)

.EXAMPLE
  .\run_daily_report_and_email.ps1 -TagHalf AM

.EXAMPLE
  .\run_daily_report_and_email.ps1 -TagHalf PM -RunPipeline:$true `
    -AttachGlob "daily_report_upbit_main.xlsx;daily_report_*.png;bt_*_summary.csv"
#>

param(
  [ValidateSet('AM','PM')]
  [string] $TagHalf = 'AM',

  [string] $Root = "D:\upbit_autotrade_starter",

  [bool]   $RunPipeline = $false,

  [string] $Subject = "[Autotrade] Upbit Main Report",

  [string] $AttachGlob = "daily_report_upbit_main.xlsx;daily_report_*.png",

  [string] $To = ""
)

$ErrorActionPreference = "Stop"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$logDir = Join-Path $Root "logs\runner"
$newLog = Join-Path $logDir ("report_email_{0}_{1}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $TagHalf)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 콘솔 + 파일에 동시 로깅
$logStream = New-Object System.IO.StreamWriter($newLog, $true, [System.Text.Encoding]::UTF8)
$logStream.AutoFlush = $true
function W($msg){ $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg; Write-Host $line; $logStream.WriteLine($line) }

Push-Location $Root
try {
  W "== RUN START == Root=$Root  Half=$TagHalf"

  # 0) venv 활성화
  $act = ".\.venv\Scripts\Activate.ps1"
  if (!(Test-Path $act)) { throw "venv Activate not found: $act" }
  W "Activating venv..."
  . $act

  # 1) (선택) 일일 파이프라인 전체 실행
  if ($RunPipeline) {
    $pipeline = ".\run_daily_pipeline.ps1"
    if (!(Test-Path $pipeline)) { throw "pipeline not found: $pipeline" }
    W "Running daily pipeline ($TagHalf)..."
    & $pipeline -TagHalf $TagHalf
    W "Pipeline done."
  } else {
    W "Skip pipeline (RunPipeline = $RunPipeline)"
  }

  # 2) 업비트 메인 리포트 생성
  $reportPy = ".\make_daily_report_upbit_main.py"
  if (!(Test-Path $reportPy)) { throw "report script missing: $reportPy" }
  W "Generating report ($TagHalf)..."
  python $reportPy --root "." --tag $TagHalf
  W "Report generated."

  # 3) 방금 생성된 daily 폴더 찾기
  $daily = Get-ChildItem ".\logs\daily" -Directory |
           Where-Object { $_.Name -like "*_$TagHalf" } |
           Sort-Object Name | Select-Object -Last 1
  if (!$daily) { throw "daily dir not found (*_$TagHalf)" }
  W ("Daily dir = {0}" -f $daily.FullName)

  # 4) 이메일 발송
  $sendPy = ".\send_report_email.py"
  if (!(Test-Path $sendPy)) { throw "send_report_email.py missing: $sendPy" }

  $args = @(
    "--daily-dir", $daily.FullName,
    "--subject",   "$Subject ($TagHalf)",
    "--body",      "자동 생성 리포트입니다. ($TagHalf)",
    "--attach-glob", $AttachGlob
  )
  if ($To -ne "") { $args += @("--to", $To) } # 선택: .env MAIL_TO override

  W "Sending email..."
  python $sendPy @args
  W "Email sent."

  W "== RUN END =="
}
catch {
  $msg = $_.Exception.Message
  Write-Warning $msg
  $logStream.WriteLine("[ERROR] " + $msg)
  exit 1
}
finally {
  $logStream.Flush()
  $logStream.Dispose()
  Pop-Location
}