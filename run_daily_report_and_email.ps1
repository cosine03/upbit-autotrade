<# 
run_daily_report_and_email.ps1
- AM/PM별 일일 리포트 생성(옵션) + 메일 발송
- 메일 전송 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- 첨부: 
  1) bt_stats_summary_merged_{AM|PM}.csv
  2) bt_breakout_only\bt_tv_events_stats_summary.csv
  3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv

.env 예 (Gmail):
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password_here    # 앱 비밀번호 (일반 비번 X)
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

$DailyDir   = Join-Path $Root "logs\daily\${DATE}_$TagHalf"
$ReportsDir = Join-Path $Root "logs\reports"
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

# ---------- Logging ----------
$EmailLog = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[{0}] {1}" -f $stamp, $msg
  $line | Tee-Object -FilePath $EmailLog -Append
}
Write-Log "== RUN START == Root=$Root  Half=$TagHalf  ==="
Write-Log "DATE            : $DATE"
Write-Log "DailyDir        : $DailyDir"

# ---------- .env loader ----------
$DotEnv = Join-Path $Root ".env"
function Load-DotEnv($path) {
  if (Test-Path -LiteralPath $path) {
    Get-Content $path | ForEach-Object {
      $line = $_.Trim()
      if (-not $line) { return }                    # 빈 줄 skip
      if ($line -match '^\s*#') { return }          # 전체 주석 줄 skip
      # 라인 끝 주석 제거
      if ($line.Contains('#')) {
        $line = $line.Split('#')[0].Trim()
        if (-not $line) { return }
      }
      # key=value
      $parts = $line.Split('=', 2)
      if ($parts.Count -ne 2) { return }
      $k = $parts[0].Trim()
      $v = $parts[1].Trim()
      # 값 감싼 따옴표 제거
      if (($v.StartsWith('"') -and $v.EndsWith('"')) -or
          ($v.StartsWith("'") -and $v.EndsWith("'"))) {
        $v = $v.Substring(1, $v.Length-2)
      }
      if ($k) { Set-Item -Path ("Env:{0}" -f $k) -Value $v }
    }
    Write-Log ".env loaded."
  } else {
    Write-Log "[WARN] .env not found at $path"
  }
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
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }

    # (필요시: 신호 분리/백테스트 호출 추가)

    # 요약 병합
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
$SMTP_HOST = if ($env:SMTP_HOST) { $env:SMTP_HOST } else { 'smtp.gmail.com' }
$SMTP_PORT = if ($env:SMTP_PORT) { [int]$env:SMTP_PORT } else { 587 }
$SMTP_USER = $env:SMTP_USER
$SMTP_PASS = $env:SMTP_PASS
$MAIL_FROM = if ($env:MAIL_FROM) { $env:MAIL_FROM } else { $SMTP_USER }
$MAIL_TO   = $env:MAIL_TO

Write-Log "SMTP_HOST=$SMTP_HOST PORT=$SMTP_PORT USER=$SMTP_USER"
Write-Log "MAIL_FROM=$MAIL_FROM"
Write-Log "MAIL_TO=$MAIL_TO"

# ---------- Compose subject/body/attachments ----------
$subject = "[Autotrade] Daily Report $DATE $TagHalf"
$body = @"
자동 생성 리포트 ($DATE $TagHalf)
첨부:
  - $(Split-Path $MergedSummary -Leaf)
  - $(Split-Path $BreakoutSummary -Leaf)
  - $(Split-Path $BoxLineSummary -Leaf)
로그: $(Resolve-Path $EmailLog)
"@

# 수신자 배열 정리 (콤마/세미콜론)
$ToList = @()
if ($MAIL_TO) {
  ($MAIL_TO -split '[,;]') | ForEach-Object {
    $addr = $_.Trim()
    if ($addr) { $ToList += $addr }
  }
}
if (-not $ToList -or $ToList.Count -eq 0) {
  Write-Log "[MAIL][ERROR] MAIL_TO empty."
  throw "MAIL_TO empty"
}

# 첨부 파일
$Attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path $p) {
    $Attachments += (Resolve-Path $p).Path
  } else {
    Write-Log "[ATTACH][WARN] not found -> $p"
  }
}

# ---------- Mail sender (System.Net.Mail; UTF-8 강제) ----------
function Send-ReportMail {
  param(
    [Parameter(Mandatory)][string]$Subject,
    [Parameter(Mandatory)][string]$Body,
    [Parameter(Mandatory)][string[]]$ToList,
    [string]$From,
    [string]$SmtpHost,
    [int]$SmtpPort = 587,
    [string]$User = $null,
    [string]$Pass = $null,
    [string[]]$Attachments = @(),
    [string]$LogPath = $null
  )

  $enc = [System.Text.Encoding]::UTF8

  $msg = New-Object System.Net.Mail.MailMessage
  if ($From) { $msg.From = $From } else { throw "MAIL_FROM is empty" }
  foreach ($to in $ToList) { [void]$msg.To.Add($to) }
  $msg.Subject         = $Subject
  $msg.SubjectEncoding = $enc
  $msg.IsBodyHtml      = $false

  # === 본문을 AlternateView로 UTF-8 + Base64 전송 (가장 호환성 좋음) ===
  $enc = [System.Text.Encoding]::UTF8
  $plainType = New-Object System.Net.Mime.ContentType "text/plain; charset=utf-8"
  $msg.IsBodyHtml = $true
  $htmlType = New-Object System.Net.Mime.ContentType "text/html; charset=utf-8"
  $alt = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($Body, [System.Text.Encoding]::UTF8, $htmlType.MediaType)
  $alt.ContentType.CharSet = "utf-8"
  $alt.TransferEncoding = [System.Net.Mime.TransferEncoding]::Base64
  ...
  $msg.AlternateViews.Clear()
  [void]$msg.AlternateViews.Add($alt)

  # 호환용으로 Body에도 동일 데이터/인코딩 지정
  $msg.Body         = $Body
  $msg.BodyEncoding = $enc
  $msg.SubjectEncoding = $enc
  if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $enc }

  # 호환용 Body 세팅
  $msg.Body         = $Body
  $msg.BodyEncoding = $enc
  if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $enc }

  # 첨부
  foreach ($p in $Attachments) {
    if ($p -and (Test-Path -LiteralPath $p)) {
      $att = New-Object System.Net.Mail.Attachment($p)
      if ($att.PSObject.Properties.Name -contains 'NameEncoding') {
        $att.NameEncoding = $enc
      }
      [void]$msg.Attachments.Add($att)
    }
  }

  $client = New-Object System.Net.Mail.SmtpClient($SmtpHost, $SmtpPort)
  $client.EnableSsl = $true
  if ($User -and $Pass) {
    $client.Credentials = New-Object System.Net.NetworkCredential($User, $Pass)
  }

  try {
    $client.Send($msg)
    if ($LogPath) { "[MAIL][OK] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Subject=$Subject" | Out-File -FilePath $LogPath -Append -Encoding UTF8 }
  } catch {
    $err = $_.Exception.Message
    if ($LogPath) { "[MAIL][ERROR] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $err" | Out-File -FilePath $LogPath -Append -Encoding UTF8 }
    throw
  } finally {
    $msg.Dispose()
    $client.Dispose()
  }
}

# ---------- Send ----------
try {
  Send-ReportMail `
    -Subject $subject `
    -Body $body `
    -ToList $ToList `
    -From $MAIL_FROM `
    -SmtpHost $SMTP_HOST `
    -SmtpPort $SMTP_PORT `
    -User $SMTP_USER `
    -Pass $SMTP_PASS `
    -Attachments $Attachments `
    -LogPath $EmailLog

  Write-Log "== DONE =="
  exit 0
} catch {
  Write-Log "[FATAL] $($_.Exception.Message)"
  exit 1
}