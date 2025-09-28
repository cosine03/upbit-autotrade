<# 
run_daily_report_and_email.ps1
- AM/PM daily report (optional pipeline run) + email send
- Logs: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- Attachments:
  1) bt_stats_summary_merged_{AM|PM}.csv
  2) bt_breakout_only\bt_tv_events_stats_summary.csv
  3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv

ENV (.env):
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password_here
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

# ---------- Paths / Date ----------
$Root       = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$DATE       = Get-Date -Format "yyyy-MM-dd"
$DailyDir   = Join-Path $Root "logs\daily\${DATE}_$TagHalf"
$ReportsDir = Join-Path $Root "logs\reports"
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

# ---------- Logging ----------
$EmailLog = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[{0}] {1}" -f $stamp, $msg
  $line  | Tee-Object -FilePath $EmailLog -Append
}
Write-Log "== RUN START == Root=$Root Half=$TagHalf =="

# ---------- .env loader ----------
function Load-DotEnv($path) {
  if (-not (Test-Path -LiteralPath $path)) { Write-Log "[WARN] .env not found: $path"; return }
  Get-Content $path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line) { return }
    if ($line -match '^\s*#') { return }
    if ($line.Contains('#')) { $line = $line.Split('#')[0].Trim(); if (-not $line) { return } }
    $parts = $line.Split('=',2)
    if ($parts.Count -ne 2) { return }
    $k = $parts[0].Trim()
    $v = $parts[1].Trim()
    if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) {
      $v = $v.Substring(1, $v.Length-2)
    }
    if ($k) { Set-Item -Path ("Env:{0}" -f $k) -Value $v }
  }
  Write-Log ".env loaded."
}
Load-DotEnv (Join-Path $Root ".env")

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

# ---------- Optional pipeline (only merge here) ----------
if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }
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
  } catch { Write-Log "[PIPE][ERROR] $($_.Exception.Message)" }
} else {
  if ((-not (Test-Path $MergedSummary)) -and (Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
    try {
      $b = Import-Csv $BreakoutSummary
      $l = Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log "merged summary saved -> $MergedSummary"
    } catch { Write-Log "[MERGE][ERROR] $($_.Exception.Message)" }
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

# ---------- helpers: CSV -> HTML table ----------
Add-Type -AssemblyName System.Web | Out-Null

function Get-HtmlTableFromCsv {
  param(
    [Parameter(Mandatory)][string]$CsvPath,
    [int]$Top = 10
  )
  if (-not (Test-Path -LiteralPath $CsvPath)) { return "<p><i>missing: $([System.Web.HttpUtility]::HtmlEncode($CsvPath))</i></p>" }
  $rows = Import-Csv $CsvPath | Select-Object -First $Top
  if (-not $rows -or $rows.Count -eq 0) { return "<p><i>no rows</i></p>" }

  $cols = $rows[0].PSObject.Properties.Name
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine("<table style='border-collapse:collapse;border:1px solid #ccc;font-family:Segoe UI,Arial,sans-serif;font-size:13px'>")
  [void]$sb.AppendLine("<thead><tr>")
  foreach ($c in $cols) { [void]$sb.AppendLine("<th style='border:1px solid #ccc;background:#f5f5f5;text-align:left;padding:6px 8px'>$( [System.Web.HttpUtility]::HtmlEncode($c) )</th>") }
  [void]$sb.AppendLine("</tr></thead><tbody>")
  foreach ($r in $rows) {
    [void]$sb.AppendLine("<tr>")
    foreach ($c in $cols) {
      $v = [string]$r.$c
      [void]$sb.AppendLine("<td style='border:1px solid #eee;padding:6px 8px;white-space:nowrap'>$( [System.Web.HttpUtility]::HtmlEncode($v) )</td>")
    }
    [void]$sb.AppendLine("</tr>")
  }
  [void]$sb.AppendLine("</tbody></table>")
  $sb.ToString()
}

# ---------- build body (plain + html) ----------
$Subject = "[Autotrade] Daily Report $DATE $TagHalf"

# Plain text (ASCII only)
$BodyText = @"
Autogenerated daily report ($DATE $TagHalf)

Attachments:
 - $(Split-Path $MergedSummary -Leaf)
 - $(Split-Path $BreakoutSummary -Leaf)
 - $(Split-Path $BoxLineSummary -Leaf)

Log: $(Resolve-Path $EmailLog)
"@

# HTML body (English headings)
$MergedTbl   = Get-HtmlTableFromCsv -CsvPath $MergedSummary -Top 10
$BreakoutTbl = Get-HtmlTableFromCsv -CsvPath $BreakoutSummary -Top 10
$BoxLineTbl  = Get-HtmlTableFromCsv -CsvPath $BoxLineSummary -Top 10

$BodyHtml = @"
<!doctype html>
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <title>Autotrade Daily Report ($DATE $TagHalf)</title>
</head>
<body style="margin:0;padding:16px;font-family:Segoe UI, Arial, sans-serif;font-size:14px;color:#111">
  <h2 style="margin:0 0 8px 0">Autotrade Daily Report ($DATE $TagHalf)</h2>

  <p style="margin:8px 0 12px 0">
    This email was generated automatically. See CSV attachments for full details.
  </p>

  <ul style="margin:0 0 16px 18px">
    <li><b>Merged:</b> $(Split-Path $MergedSummary -Leaf)</li>
    <li><b>Breakout:</b> $(Split-Path $BreakoutSummary -Leaf)</li>
    <li><b>Box+Line:</b> $(Split-Path $BoxLineSummary -Leaf)</li>
  </ul>

  <h3 style="margin:16px 0 8px 0">Merged Summary (Top 10)</h3>
  $MergedTbl

  <h3 style="margin:24px 0 8px 0">Breakout Summary (Top 10)</h3>
  $BreakoutTbl

  <h3 style="margin:24px 0 8px 0">Box+Line Summary (Top 10)</h3>
  $BoxLineTbl

  <p style="margin-top:16px;color:#666">
    Log file: $(Resolve-Path $EmailLog)
  </p>
</body>
</html>
"@

# ---------- recipients / attachments ----------
$ToList = @()
if ($MAIL_TO) { $MAIL_TO.Split(',;') | ForEach-Object { $a=$_.Trim(); if ($a) { $ToList += $a } } }
if (-not $ToList -or $ToList.Count -eq 0) { Write-Log "[MAIL][ERROR] MAIL_TO empty."; throw "MAIL_TO empty" }

$Attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path $p) { $Attachments += (Resolve-Path $p).Path } else { Write-Log "[ATTACH][WARN] not found -> $p" }
}

# ---------- mail sender (multipart: text/plain + text/html) ----------
function Send-ReportMail {
  param(
    [Parameter(Mandatory)][string]$Subject,
    [Parameter(Mandatory)][string]$BodyText,  # ASCII plain text
    [Parameter(Mandatory)][string]$BodyHtml,  # UTF-8 HTML
    [Parameter(Mandatory)][string[]]$ToList,
    [Parameter(Mandatory)][string]$From,
    [Parameter(Mandatory)][string]$SmtpHost,
    [int]$SmtpPort = 587,
    [string]$User = $null,
    [string]$Pass = $null,
    [string[]]$Attachments = @(),
    [string]$LogPath = $null
  )

  $utf8  = [System.Text.Encoding]::UTF8
  $ascii = [System.Text.Encoding]::ASCII

  $msg = New-Object System.Net.Mail.MailMessage
  $msg.From = $From
  foreach ($to in $ToList) { [void]$msg.To.Add($to) }

  # subject is ASCII to be extra-safe
  $msg.Subject         = $Subject
  $msg.SubjectEncoding = $ascii
  if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $ascii }

  # Alternate views
  $altText = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyText, $ascii, "text/plain")
  $altText.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable

  $altHtml = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyHtml, $utf8, "text/html")
  $altHtml.TransferEncoding = [System.Net.Mime.TransferEncoding]::Base64  # many KR portals like this

  $msg.AlternateViews.Clear()
  $msg.AlternateViews.Add($altText)
  $msg.AlternateViews.Add($altHtml)

  # also set Body for legacy clients
  $msg.IsBodyHtml  = $false
  $msg.Body        = $BodyText
  $msg.BodyEncoding = $ascii

  foreach ($p in $Attachments) {
    if ($p -and (Test-Path -LiteralPath $p)) {
      $att = New-Object System.Net.Mail.Attachment($p)
      $msg.Attachments.Add($att) | Out-Null
    }
  }

  $client = New-Object System.Net.Mail.SmtpClient($SmtpHost, $SmtpPort)
  $client.EnableSsl = $true
  if ($User -and $Pass) { $client.Credentials = New-Object System.Net.NetworkCredential($User, $Pass) }

  try {
    $client.Send($msg)
    if ($LogPath) { "[MAIL][OK] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Subject=$Subject" | Out-File -FilePath $LogPath -Append -Encoding UTF8 }
  } catch {
    $err = $_.Exception.Message
    if ($LogPath) { "[MAIL][ERROR] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $err" | Out-File -FilePath $LogPath -Append -Encoding UTF8 }
    throw
  } finally {
    $msg.Dispose(); $client.Dispose()
  }
}

# ---------- send ----------
try {
  Send-ReportMail `
    -Subject  $Subject `
    -BodyText $BodyText `
    -BodyHtml $BodyHtml `
    -ToList   $ToList `
    -From     $MAIL_FROM `
    -SmtpHost $SMTP_HOST `
    -SmtpPort $SMTP_PORT `
    -User     $SMTP_USER `
    -Pass     $SMTP_PASS `
    -Attachments $Attachments `
    -LogPath  $EmailLog
  Write-Log "== DONE =="
  exit 0
} catch {
  Write-Log "[FATAL] $($_.Exception.Message)"
  exit 1
}