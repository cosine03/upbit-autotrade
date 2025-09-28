<# 
run_daily_report_and_email.ps1
- AM/PM별 일일 리포트 생성(옵션) + 메일 발송
- 메일 전송 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- 첨부: 
  1) bt_stats_summary_merged_{AM|PM}.csv
  2) bt_breakout_only\bt_tv_events_stats_summary.csv
  3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv

필수: 루트 경로(.env, logs, scripts)가 이 스크립트와 같은 폴더에 있다고 가정.
.Gmail 사용 시 .env 예:
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password_here   # 앱 비밀번호 (일반 비번 X)
MAIL_FROM=you@gmail.com
MAIL_TO=first@example.com,second@example.com
#>

param(
  [ValidateSet("AM","PM")]
  [string]$TagHalf = "AM",
  [switch]$RunPipeline = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------- Path & Date ----------
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$DATE = Get-Date -Format "yyyy-MM-dd"
$DailyDir = Join-Path $Root "logs\daily\${DATE}_$TagHalf"
$ReportsDir = Join-Path $Root "logs\reports"
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

# ---------- Logging ----------
$EmailLog = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line = "[{0}] {1}" -f $stamp, $msg
  $line | Tee-Object -FilePath $EmailLog -Append
}

Write-Log "== RUN START == Root=$Root  Half=$TagHalf  ==="
Write-Log "DATE            : $DATE"
Write-Log "DailyDir        : $DailyDir"

# ---------- .env loader ----------
$DotEnv = Join-Path $Root ".env"
function Load-DotEnv($path) {
  if (-not (Test-Path $path)) { 
    Write-Log ".env not found at $path (continuing; mail may fail)"
    return
  }
  Get-Content $path | ForEach-Object {
    $line = $_.Trim()
    if ($line -match "^\s*#") { return }           # comment
    if ($line -notmatch "=")  { return }           # invalid
    $k,$v = $line -split "=",2
    $k = $k.Trim()
    $v = $v.Trim().Trim("'`"").Trim()              # strip quotes if any
    if ($k) { $env:$k = $v }
  }
  Write-Log ".env loaded."
}
Load-DotEnv $DotEnv

# ---------- Key files ----------
$SignalsTV        = Join-Path $Root "logs\signals_tv.csv"
$SignalsBreakout  = Join-Path $DailyDir "signals_breakout_only.csv"
$SignalsBoxLine   = Join-Path $DailyDir "signals_boxin_linebreak.csv"
$DynParams        = Join-Path $DailyDir "dynamic_params.json"
$BtDirBreakout    = Join-Path $DailyDir "bt_breakout_only"
$BtDirBoxLine     = Join-Path $DailyDir "bt_boxin_linebreak"
$BreakoutSummary  = Join-Path $BtDirBreakout "bt_tv_events_stats_summary.csv"
$BoxLineSummary   = Join-Path $BtDirBoxLine  "bt_tv_events_stats_summary.csv"
$MergedSummary    = Join-Path $DailyDir ("bt_stats_summary_merged_{0}.csv" -f $TagHalf)

Write-Log "SignalsTV       : $SignalsTV"
Write-Log "SignalsBreakout : $SignalsBreakout"
Write-Log "SignalsBoxLine  : $SignalsBoxLine"
Write-Log "DynParamsJson   : $DynParams"
Write-Log "BtDirBreakout   : $BtDirBreakout"
Write-Log "BtDirBoxLine    : $BtDirBoxLine"
Write-Log "MergedSummary   : $MergedSummary"

# ---------- Optional: run pipeline before email ----------
if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    # (1) 일일 폴더 준비
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }

    # (2) 시그널 분리/가공 (이미 별도 ps1이 있다면 그걸 호출해도 됨)
    # 예시: 기존에 사용하던 Python/PS 파이프라인 호출
    # python .\label_events_side.py "$SignalsTV" --out "$DailyDir\signals_labeled.csv"
    # python .\filter_breakout_only.py "$DailyDir\signals_labeled.csv" --out "$SignalsBreakout" --dist-max 0.000188
    # python .\filter_boxin_linebreak.py "$DailyDir\signals_labeled.csv" --out "$SignalsBoxLine" --dist-max 0.000188

    # (3) 백테스트 실행 (기존에 쓰던 커맨드 호출)
    # python .\backtest_tv_events_mp.py "$SignalsBreakout"  --timeframe 15m --expiries 0.5h,1h,2h --tp 1.75 --sl 0.7 --fee 0.001 --entry prev_close --dist-max 0.000188 --outdir "$BtDirBreakout"
    # python .\backtest_tv_events_mp.py "$SignalsBoxLine"   --timeframe 15m --expiries 0.5h,1h,2h --tp 1.75 --sl 0.7 --fee 0.001 --entry prev_close --dist-max 0.000188 --outdir "$BtDirBoxLine"

    # (4) 요약 병합 (PowerShell 내에서 간단히 합치기)
    if ((Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
      $b = Import-Csv $BreakoutSummary
      $l = Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log "merged summary saved -> $MergedSummary"
    } else {
      Write-Log "[PIPE][WARN] summary files missing; skip merge."
    }
    Write-Log "[PIPE] done"
  } catch {
    Write-Log "[PIPE][ERROR] $($_.Exception.Message)"
  }
} else {
  # 파이프라인을 돌리지 않는 경우: merged 가 없으면 만들어 준다(가능한 경우)
  if ((-not (Test-Path $MergedSummary)) -and (Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
    try {
      $b = Import-Csv $BreakoutSummary
      $l = Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log "merged summary saved -> $MergedSummary"
    } catch {
      Write-Log "[MERGE][ERROR] $($_.Exception.Message)"
    }
  }
}

# ---------- Email config from .env ----------
$SMTP_HOST = $env:SMTP_HOST
$SMTP_PORT = [int]($env:SMTP_PORT ? $env:SMTP_PORT : 587)
$SMTP_USER = $env:SMTP_USER
$SMTP_PASS = $env:SMTP_PASS
$MAIL_FROM = $env:MAIL_FROM
$MAIL_TO   = $env:MAIL_TO

Write-Log "SMTP_HOST=$SMTP_HOST PORT=$SMTP_PORT USER=$SMTP_USER"
Write-Log "MAIL_FROM=$MAIL_FROM"
Write-Log "MAIL_TO=$MAIL_TO"

# ---------- Compose email ----------
$subject = "[Autotrade] Daily Report $DATE $TagHalf"
$body = @"
자동 생성 리포트 ($DATE $TagHalf)

첨부:
  - $(Split-Path $MergedSummary -Leaf)
  - $(Split-Path $BreakoutSummary -Leaf)
  - $(Split-Path $BoxLineSummary -Leaf)

로그: $(Resolve-Path $EmailLog)
"@

# 수신자 배열 정리(콤마/세미콜론 구분 허용)
$toList = @()
if ($MAIL_TO) {
  $MAIL_TO.Split(',;') | ForEach-Object {
    $addr = $_.Trim()
    if ($addr) { $toList += $addr }
  }
}

# 첨부 파일 존재 확인
$attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path $p) {
    $attachments += (Resolve-Path $p).Path
  } else {
    Write-Log "[ATTACH][WARN] not found -> $p"
  }
}

# ---------- Send via System.Net.Mail ----------
try {
  if (-not $SMTP_HOST -or -not $SMTP_USER -or -not $SMTP_PASS -or -not $MAIL_FROM -or $toList.Count -eq 0) {
    throw "Missing SMTP settings or recipients. Check .env"
  }

  Add-Type -AssemblyName System.Net.Mail
  $mail = New-Object System.Net.Mail.MailMessage
  $mail.From = $MAIL_FROM
  foreach ($rcpt in $toList) { $mail.To.Add($rcpt) }
  $mail.Subject = $subject
  $mail.Body = $body
  $mail.IsBodyHtml = $false

  foreach ($path in $attachments) {
    $att = New-Object System.Net.Mail.Attachment($path)
    $mail.Attachments.Add($att) | Out-Null
  }

  $client = New-Object System.Net.Mail.SmtpClient($SMTP_HOST, $SMTP_PORT)
  $client.EnableSsl = $true
  $client.Credentials = New-Object System.Net.NetworkCredential($SMTP_USER, $SMTP_PASS)

  Write-Log "[MAIL] sending..."
  $client.Send($mail)
  Write-Log "[MAIL] sent OK -> $($toList -join ', ')"
}
catch {
  Write-Log "[MAIL][ERROR] $($_.Exception.Message)"
  exit 1
}
finally {
  if ($mail) { $mail.Dispose() }
  if ($client) { $client.Dispose() }
}

Write-Log "== DONE =="
exit 0