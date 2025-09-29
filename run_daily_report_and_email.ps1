<# 
run_daily_report_and_email.ps1
- AM/PM 리포트 생성(+옵션 파이프라인) 후 HTML 메일 발송
- 본문: Merged / Breakout / Box-Line 각 Top10 (퍼센트 P2로 표기)
- 카카오 알림 미리보기: text/plain + text/html 멀티파트
- 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
#>

param(
  [ValidateSet("AM","PM")]
  [string]$TagHalf = "AM",
  [switch]$RunPipeline = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------- 경로/날짜 ----------
$Root       = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$DATE       = Get-Date -Format "yyyy-MM-dd"
$DailyDir   = Join-Path $Root "logs\daily\${DATE}_$TagHalf"
$ReportsDir = Join-Path $Root "logs\reports"
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

# ---------- 로깅 ----------
$EmailLog = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[{0}] {1}" -f $stamp, $msg
  $line | Tee-Object -FilePath $EmailLog -Append | Out-Null
}
Write-Log "== RUN START == Root=$Root Half=$TagHalf =="

# ---------- .env ----------
$DotEnv = Join-Path $Root ".env"
if (Test-Path -LiteralPath $DotEnv) {
  Get-Content $DotEnv | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line -match '^\s*#') { return }
    if ($line.Contains('#')) { $line = $line.Split('#')[0].Trim(); if (-not $line) { return } }
    $kv = $line.Split('=',2); if ($kv.Count -ne 2) { return }
    $k=$kv[0].Trim(); $v=$kv[1].Trim()
    if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) { $v=$v.Substring(1,$v.Length-2) }
    if ($k) { Set-Item -Path ("Env:{0}" -f $k) -Value $v }
  }
  Write-Log ".env loaded."
} else {
  Write-Log "[WARN] .env not found: $DotEnv"
}

# ---------- 파일 ----------
$BtDirBreakout   = Join-Path $DailyDir "bt_breakout_only"
$BtDirBoxLine    = Join-Path $DailyDir "bt_boxin_linebreak"
$BreakoutSummary = Join-Path $BtDirBreakout "bt_tv_events_stats_summary.csv"
$BoxLineSummary  = Join-Path $BtDirBoxLine  "bt_tv_events_stats_summary.csv"
$MergedSummary   = Join-Path $DailyDir ("bt_stats_summary_merged_{0}.csv" -f $TagHalf)

Write-Log ("[CHECK] exists merged={0} breakout={1} boxline={2}" -f `
  ($(if (Test-Path $MergedSummary) {1}else{0}),
   $(if (Test-Path $BreakoutSummary){1}else{0}),
   $(if (Test-Path $BoxLineSummary) {1}else{0})))

# ---------- 파이프라인(옵션) ----------
if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }
    if ((Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
      $b=Import-Csv $BreakoutSummary
      $l=Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b+$l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
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
      $b=Import-Csv $BreakoutSummary
      $l=Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b+$l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log "merged summary saved -> $MergedSummary"
    } catch { Write-Log "[MERGE][ERROR] $($_.Exception.Message)" }
  }
}

# ---------- 메일 설정 ----------
$SMTP_HOST = if ($env:SMTP_HOST) { $env:SMTP_HOST } else { 'smtp.gmail.com' }
$SMTP_PORT = if ($env:SMTP_PORT) { [int]$env:SMTP_PORT } else { 587 }
$SMTP_USER = $env:SMTP_USER
$SMTP_PASS = $env:SMTP_PASS
$MAIL_FROM = if ($env:MAIL_FROM) { $env:MAIL_FROM } else { $SMTP_USER }
$MAIL_TO   = $env:MAIL_TO

# ---------- 표 HTML 유틸 ----------
Add-Type -AssemblyName System.Web
function New-TopHtml {
  param(
    [Parameter(Mandatory)][string]$CsvPath,
    [int]$Top = 10,
    [string]$Title = ""
  )
  if (-not (Test-Path -LiteralPath $CsvPath)) { return "<p style='color:#999;'>${Title}: file not found</p>" }
  $rows = Import-Csv $CsvPath | Select-Object event,expiry_h,trades,win_rate,avg_net,median_net,total_net,strategy
  if (-not $rows) { return "<p style='color:#999;'>${Title}: no rows</p>" }

  $rows = $rows | Select-Object `
    event,expiry_h,trades,
    @{n='win_rate';e={[double]$_.win_rate}},
    @{n='avg_net';e={[double]$_.avg_net}},
    @{n='median_net';e={[double]$_.median_net}},
    @{n='total_net';e={[double]$_.total_net}},
    strategy |
    Sort-Object -Property total_net -Descending | Select-Object -First $Top

  $cols = 'event','expiry_h','trades','win_rate','avg_net','median_net','total_net','strategy'
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine("<h3 style='margin:16px 0 8px'>$([System.Web.HttpUtility]::HtmlEncode($Title))</h3>")
  [void]$sb.AppendLine("<table style='border-collapse:collapse;width:860px'>")
  [void]$sb.AppendLine("<thead><tr>")
  foreach ($c in $cols) {
    [void]$sb.AppendLine("<th style='padding:6px 8px;border:1px solid #ddd;background:#f7f7f7;text-align:left;'>$([System.Web.HttpUtility]::HtmlEncode($c))</th>")
  }
  [void]$sb.AppendLine("</tr></thead><tbody>")
  foreach ($r in $rows) {
    [void]$sb.AppendLine("<tr>")
    foreach ($c in $cols) {
      $v = $r.$c
      if ($c -in @('win_rate','avg_net','median_net','total_net')) {
        $v = ([double]$v).ToString("P2", [System.Globalization.CultureInfo]::InvariantCulture)
      }
      [void]$sb.AppendLine("<td style='padding:6px 8px;border:1px solid #ddd;'>$([System.Web.HttpUtility]::HtmlEncode([string]$v))</td>")
    }
    [void]$sb.AppendLine("</tr>")
  }
  [void]$sb.AppendLine("</tbody></table>")
  $sb.ToString()
}

# ---------- 본문 ----------
$Subject = "[Autotrade] Daily Report $DATE $TagHalf"

$MergedHtml   = New-TopHtml -CsvPath $MergedSummary   -Top 10 -Title "Merged Summary (Top 10)"
$BreakoutHtml = New-TopHtml -CsvPath $BreakoutSummary -Top 10 -Title "Breakout Summary (Top 10)"
$BoxLineHtml  = New-TopHtml -CsvPath $BoxLineSummary  -Top 10 -Title "Box-Line Summary (Top 10)"

$PreviewText = @"
Autotrade Daily Report ($DATE $TagHalf)
Attachments: 3
This is an HTML email with CSV attachments.
"@.Trim()

$BodyHtml = @"
<!doctype html>
<html>
<head>
<meta charset="UTF-8" />
<title>$([System.Web.HttpUtility]::HtmlEncode($Subject))</title>
<style>
  body{font-family:Segoe UI,Arial,sans-serif;font-size:14px;color:#e8e8e8;background:#121212}
  a{color:#8ab4f8}
  .wrap{max-width:900px;margin:0 auto;padding:16px 8px}
  h2{margin:0 0 12px}
  p{margin:6px 0}
  table{border-collapse:collapse;width:100%;margin:8px 0}
  th,td{border:1px solid #2a2a2a;padding:6px 8px}
  thead th{background:#1f1f1f}
  .muted{color:#bdbdbd}
</style>
</head>
<body>
<div class="wrap">
  <h2>Autotrade Daily Report ($DATE $TagHalf)</h2>
  <p class="muted">Attachments: 3</p>
  $MergedHtml
  $BreakoutHtml
  $BoxLineHtml
  <p class="muted">Log: $([System.Web.HttpUtility]::HtmlEncode((Resolve-Path $EmailLog).Path))</p>
</div>
</body>
</html>
"@

# ---------- 수신자/첨부 ----------
$ToList = @()
if ($MAIL_TO) { $MAIL_TO.Split(',;') | ForEach-Object { $a=$_.Trim(); if ($a) { $ToList += $a } } }
if (-not $ToList) { Write-Log "[MAIL][ERROR] MAIL_TO empty."; throw "MAIL_TO empty" }

$Attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path -LiteralPath $p) { $Attachments += (Resolve-Path $p).Path } else { Write-Log "[ATTACH][WARN] not found -> $p" }
}

# ---------- 메일 발송 ----------
$enc  = [System.Text.Encoding]::UTF8
$msg  = New-Object System.Net.Mail.MailMessage
$msg.From = $MAIL_FROM
foreach ($t in $ToList) { [void]$msg.To.Add($t) }
$msg.Subject         = $Subject
$msg.SubjectEncoding = $enc
$msg.IsBodyHtml      = $false

$altText = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($PreviewText, $enc, "text/plain")
$altText.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
$altHtml = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyHtml, $enc, "text/html")
$altHtml.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
$msg.AlternateViews.Add($altText)
$msg.AlternateViews.Add($altHtml)

foreach ($p in $Attachments) {
  $att = New-Object System.Net.Mail.Attachment($p)
  if ($att.PSObject.Properties.Name -contains 'NameEncoding') { $att.NameEncoding = $enc }
  $msg.Attachments.Add($att) | Out-Null
}

$smtp = New-Object System.Net.Mail.SmtpClient($SMTP_HOST, $SMTP_PORT)
$smtp.EnableSsl = $true
if ($SMTP_USER -and $SMTP_PASS) { $smtp.Credentials = New-Object System.Net.NetworkCredential($SMTP_USER, $SMTP_PASS) }

try {
  $smtp.Send($msg)
  Write-Log ("[MAIL][OK] {0} sent to {1}" -f $Subject, ($ToList -join ", "))
} catch {
  Write-Log "[MAIL][ERROR] $($_.Exception.Message)"
  throw
} finally {
  $msg.Dispose()
  $smtp.Dispose()
}

Write-Log "== DONE =="
exit 0