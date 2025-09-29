<#
run_daily_report_and_email.ps1
- AM/PM별 일일 리포트 생성(옵션) + 메일 발송
- 메일 전송 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- 첨부:
  1) bt_stats_summary_merged_{AM|PM}.csv
  2) bt_breakout_only\bt_tv_events_stats_summary.csv
  3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv
필수: 루트 경로(.env, logs, scripts)가 이 스크립트와 같은 폴더에 있다고 가정.
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
Write-Log "== RUN START == Root=$Root Half=$TagHalf =="

# ---------- .env loader ----------
$DotEnv = Join-Path $Root ".env"
function Load-DotEnv($path) {
  if (Test-Path -LiteralPath $path) {
    Get-Content $path | ForEach-Object {
      $line = $_.Trim()
      if (-not $line) { return }
      if ($line -match '^\s*#') { return }
      if ($line.Contains('#')) {
        $line = $line.Split('#')[0].Trim()
        if (-not $line) { return }
      }
      $parts = $line.Split('=', 2)
      if ($parts.Count -ne 2) { return }
      $k = $parts[0].Trim()
      $v = $parts[1].Trim()
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
$SignalsBreakout = Join-Path $DailyDir "signals_breakout_only.csv"
$SignalsBoxLine  = Join-Path $DailyDir "signals_boxin_linebreak.csv"
$BtDirBreakout   = Join-Path $DailyDir "bt_breakout_only"
$BtDirBoxLine    = Join-Path $DailyDir "bt_boxin_linebreak"
$BreakoutSummary = Join-Path $BtDirBreakout "bt_tv_events_stats_summary.csv"
$BoxLineSummary  = Join-Path $BtDirBoxLine  "bt_tv_events_stats_summary.csv"
$MergedSummary   = Join-Path $DailyDir ("bt_stats_summary_merged_{0}.csv" -f $TagHalf)

# ---------- Optional: run pipeline ----------
if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    if ((Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
      $b = Import-Csv $BreakoutSummary
      $l = Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log "merged summary saved -> $MergedSummary"
    }
    Write-Log "[PIPE] done"
  } catch {
    Write-Log "[PIPE][ERROR] $($_.Exception.Message)"
  }
}

# ---------- Email config ----------
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
$Subject = "[Autotrade] Daily Report $DATE $TagHalf"

# 첨부 파일 목록
$AttachList = @()
foreach ($p in @($MergedSummary,$BreakoutSummary,$BoxLineSummary)) {
  if (Test-Path $p) { $AttachList += (Resolve-Path $p).Path }
}
$AttachCount = $AttachList.Count

# HTML 본문에 테이블 넣기
function Get-TopHtmlTable($CsvPath,$Top=10) {
  if (-not (Test-Path $CsvPath)) { return "<p>[WARN] $CsvPath not found</p>" }
  $rows = Import-Csv $CsvPath | Select-Object -First $Top
  if (-not $rows) { return "<p>[WARN] No rows in $CsvPath</p>" }
  $cols = $rows[0].PSObject.Properties.Name
  $sb = New-Object -TypeName System.Text.StringBuilder
  [void]$sb.AppendLine("<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;width:100%'>")
  [void]$sb.AppendLine('<thead><tr>')
  foreach ($c in $cols) { [void]$sb.AppendLine("<th style='background:#f2f2f2;text-align:left;'>$c</th>") }
  [void]$sb.AppendLine('</tr></thead><tbody>')
  foreach ($r in $rows) {
    [void]$sb.AppendLine('<tr>')
    foreach ($c in $cols) { [void]$sb.AppendLine("<td>$($r.$c)</td>") }
    [void]$sb.AppendLine('</tr>')
  }
  [void]$sb.AppendLine('</tbody></table>')
  $sb.ToString()
}

$TableHtml = Get-TopHtmlTable -CsvPath $MergedSummary -Top 10
$BodyHtml = @"
<!doctype html>
<html>
<head><meta charset="UTF-8" /></head>
<body>
  <h2>Autotrade Daily Report ($DATE $TagHalf)</h2>
  <p>Attachments: $AttachCount</p>
  $TableHtml
</body>
</html>
"@
$BodyText = "Autotrade Daily Report ($DATE $TagHalf)`nAttachments: $AttachCount"

# ---------- Send email ----------
$enc = [System.Text.Encoding]::UTF8
$msg = New-Object System.Net.Mail.MailMessage
$msg.From = $MAIL_FROM
foreach ($t in $MAIL_TO.Split(',;')) { if ($t.Trim()) { [void]$msg.To.Add($t.Trim()) } }
$msg.Subject         = $Subject
$msg.SubjectEncoding = $enc
$msg.IsBodyHtml      = $true

# plain
$altText = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyText,$enc,"text/plain")
$altText.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
$msg.AlternateViews.Add($altText)

# html
$altHtml = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyHtml,$enc,"text/html")
$altHtml.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
$msg.AlternateViews.Add($altHtml)

$msg.Body         = $BodyHtml
$msg.BodyEncoding = $enc
if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $enc }

foreach ($fp in $AttachList) {
  $att = New-Object System.Net.Mail.Attachment($fp)
  if ($att.PSObject.Properties.Name -contains 'NameEncoding') { $att.NameEncoding = $enc }
  [void]$msg.Attachments.Add($att)
}

$smtp = New-Object System.Net.Mail.SmtpClient($SMTP_HOST,[int]$SMTP_PORT)
$smtp.EnableSsl = $true
$smtp.Credentials = New-Object System.Net.NetworkCredential($SMTP_USER,$SMTP_PASS)

try {
  $smtp.Send($msg)
  Write-Log "[MAIL][OK] $Subject sent to $MAIL_TO"
} catch {
  Write-Log "[MAIL][ERROR] $($_.Exception.Message)"
} finally {
  $msg.Dispose()
  $smtp.Dispose()
}