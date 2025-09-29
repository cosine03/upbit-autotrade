<# 
run_daily_report_and_email.ps1
- AM/PM별 일일 리포트 생성(옵션) + 메일 발송
- 메일 전송 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- 첨부: 
  1) bt_stats_summary_merged_{AM|PM}.csv
  2) bt_breakout_only\bt_tv_events_stats_summary.csv
  3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv

.env 예(Gmail):
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

# ---------- Path & Date ----------
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$DATE = Get-Date -Format "yyyy-MM-dd"

$DailyDir    = Join-Path $Root "logs\daily\${DATE}_$TagHalf"
$ReportsDir  = Join-Path $Root "logs\reports"
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

# ---------- Logging ----------
$EmailLog = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[{0}] {1}" -f $stamp, $msg
  $line | Tee-Object -FilePath $EmailLog -Append
}

Write-Log ("== RUN START == Root={0} Half={1} ==" -f $Root, $TagHalf)

# ---------- .env loader ----------
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
    Write-Log ("[WARN] .env not found at {0}" -f $path)
  }
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

Write-Log ("[CHECK] exists merged={0} breakout={1} boxline={2}" -f `
    ($(if (Test-Path $MergedSummary) {1} else {0}), `
     $(if (Test-Path $BreakoutSummary) {1} else {0}), `
     $(if (Test-Path $BoxLineSummary) {1} else {0})))

# ---------- Optional pipeline (merge only; 외부 호출은 필요시 추가) ----------
if ($RunPipeline) {
  Write-Log "[PIPE] start"
  try {
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }
    if ((Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
      $b = Import-Csv $BreakoutSummary
      $l = Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log ("merged summary saved -> {0}" -f $MergedSummary)
    } else {
      Write-Log "[PIPE][WARN] summary files missing; skip merge."
    }
  } catch {
    Write-Log ("[PIPE][ERROR] {0}" -f $_.Exception.Message)
  }
  Write-Log "[PIPE] done"
} else {
  if ((-not (Test-Path $MergedSummary)) -and (Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
    try {
      $b = Import-Csv $BreakoutSummary
      $l = Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log ("merged summary saved -> {0}" -f $MergedSummary)
    } catch {
      Write-Log ("[MERGE][ERROR] {0}" -f $_.Exception.Message)
    }
  }
}

# ---------- Email config ----------
$SMTP_HOST = if ($env:SMTP_HOST) { $env:SMTP_HOST } else { 'smtp.gmail.com' }
$SMTP_PORT = if ($env:SMTP_PORT) { [int]$env:SMTP_PORT } else { 587 }
$SMTP_USER = $env:SMTP_USER
$SMTP_PASS = $env:SMTP_PASS
$MAIL_FROM = if ($env:MAIL_FROM) { $env:MAIL_FROM } else { $SMTP_USER }
$MAIL_TO   = $env:MAIL_TO

Write-Log ("SMTP_HOST={0} PORT={1} USER={2}" -f $SMTP_HOST, $SMTP_PORT, $SMTP_USER)
Write-Log ("MAIL_FROM={0}" -f $MAIL_FROM)
Write-Log ("MAIL_TO={0}" -f $MAIL_TO)

# ---------- Helpers: CSV -> HTML TopN ----------
function Get-TopHtmlTable {
  param(
    [Parameter(Mandatory)][string]$CsvPath,
    [int]$Top = 10,
    [string]$Title = ""
  )
  if (-not (Test-Path $CsvPath)) { return "" }
  $rows = Import-Csv $CsvPath
  if (-not $rows -or $rows.Count -eq 0) { return "" }

  # 숫자형 보장 및 정렬(총합 내림차순)
  foreach ($r in $rows) {
    $r.trades     = [int]$r.trades
    $r.win_rate   = [double]$r.win_rate
    $r.avg_net    = [double]$r.avg_net
    $r.total_net  = [double]$r.total_net
    $r.expiry_h   = [double]$r.expiry_h
  }
  $topRows = $rows | Sort-Object -Property {[double]$_.total_net} -Descending | Select-Object -First $Top

  # 컬럼 구성 및 포맷
  $enc = [System.Web.HttpUtility]
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine("<table style='border-collapse:collapse;width:100%;font-family:Segoe UI,Arial,sans-serif;font-size:13px;'>")
  if ($Title) {
    [void]$sb.AppendLine("<caption style='text-align:left;font-weight:bold;padding:6px 0;'>$($enc::HtmlEncode($Title))</caption>")
  }
  [void]$sb.AppendLine("<thead><tr>")
  $headers = @("Strategy","Event","Expiry(h)","Trades","Win%","Avg Net","Total Net")
  foreach ($h in $headers) {
    [void]$sb.AppendLine("<th style='background:#f2f2f2;text-align:left;padding:6px;border:1px solid #ddd;'>$($enc::HtmlEncode($h))</th>")
  }
  [void]$sb.AppendLine("</tr></thead><tbody>")

  foreach ($r in $topRows) {
    $strategy = $r.strategy
    $event    = $r.event
    $exp      = ("{0:0.##}" -f [double]$r.expiry_h)
    $trades   = ("{0}" -f [int]$r.trades)
    $winp     = ("{0:0.00}%" -f (([double]$r.win_rate)*100.0))
    $avgp     = ("{0:0.00}%" -f (([double]$r.avg_net)*100.0))
    $totalp   = ("{0:0.00}%" -f (([double]$r.total_net)*100.0))

    [void]$sb.AppendLine("<tr>")
    foreach ($val in @($strategy,$event,"$exp","$trades","$winp","$avgp","$totalp")) {
      [void]$sb.AppendLine("<td style='padding:6px;border:1px solid #ddd;'>$($enc::HtmlEncode([string]$val))</td>")
    }
    [void]$sb.AppendLine("</tr>")
  }
  [void]$sb.AppendLine("</tbody></table>")
  $sb.ToString()
}

# ---------- Build HTML body (3 tables) ----------
function Build-ReportHtml {
  param(
    [string]$MergedCsv,
    [string]$BreakoutCsv,
    [string]$BoxLineCsv,
    [string]$DateStr,
    [string]$HalfTag
  )
  $sections = @()

  try {
    if (Test-Path $MergedCsv) {
      $t = Get-TopHtmlTable -CsvPath $MergedCsv -Top 10 -Title "TOP 10 (Merged)"
      if ($t) { $sections += $t } else { Write-Log ("[HTML] {0}: empty" -f "merged") }
    } else {
      Write-Log ("[HTML] {0}: file missing -> {1}" -f "merged", $MergedCsv)
    }

    if (Test-Path $BreakoutCsv) {
      $t = Get-TopHtmlTable -CsvPath $BreakoutCsv -Top 10 -Title "TOP 10 (Breakout Only)"
      if ($t) { $sections += $t } else { Write-Log ("[HTML] {0}: empty" -f "breakout") }
    } else {
      Write-Log ("[HTML] {0}: file missing -> {1}" -f "breakout", $BreakoutCsv)
    }

    if (Test-Path $BoxLineCsv) {
      $t = Get-TopHtmlTable -CsvPath $BoxLineCsv -Top 10 -Title "TOP 10 (Boxin + Linebreak)"
      if ($t) { $sections += $t } else { Write-Log ("[HTML] {0}: empty" -f "boxline") }
    } else {
      Write-Log ("[HTML] {0}: file missing -> {1}" -f "boxline", $BoxLineCsv)
    }
  } catch {
    Write-Log ("[HTML][ERROR] {0}" -f $_.Exception.Message)
  }

  $joined = ($sections -join "<div style='height:12px;'></div>")

  @"
<!doctype html>
<html>
<head>
<meta charset="UTF-8" />
<title>Autotrade Daily Report $DateStr $HalfTag</title>
</head>
<body style="margin:0;padding:0;">
  <div style="max-width:960px;margin:0 auto;padding:12px 14px;font-family:Segoe UI,Arial,sans-serif;">
    <h2 style="margin:0 0 8px;">Autotrade Daily Report - $DateStr $HalfTag</h2>
    <p style="margin:0 0 12px;">Auto-generated summary. See attachments for full CSVs.</p>
    $joined
  </div>
</body>
</html>
"@
}

# ---------- Compose subject/body/attachments ----------
$subject = "[Autotrade] Daily Report $DATE $TagHalf"
$plainBody = "Autotrade Daily Report ($DATE $TagHalf) - Please see the HTML body and attachments."

# HTML 본문 생성 & 디버그 저장
$BodyHtml = Build-ReportHtml -MergedCsv $MergedSummary -BreakoutCsv $BreakoutSummary -BoxLineCsv $BoxLineSummary -DateStr $DATE -HalfTag $TagHalf
$debugHtmlPath = Join-Path $ReportsDir ("email_{0}_{1}.html" -f $DATE, $TagHalf)
$BodyHtml | Out-File -FilePath $debugHtmlPath -Encoding UTF8

# 수신자 배열
$ToList = @()
if ($MAIL_TO) {
  $MAIL_TO.Split(',;') | ForEach-Object {
    $addr = $_.Trim()
    if ($addr) { $ToList += $addr }
  }
}
if (-not $ToList -or $ToList.Count -eq 0) {
  Write-Log "[MAIL][ERROR] MAIL_TO empty."
  throw "MAIL_TO empty"
}

# 첨부
$Attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path $p) {
    $Attachments += (Resolve-Path $p).Path
  } else {
    Write-Log ("[ATTACH][WARN] not found -> {0}" -f $p)
  }
}

# ---------- Mail sender (System.Net.Mail; UTF-8 & AltViews) ----------
function Send-ReportMail {
  param(
    [Parameter(Mandatory)][string]$Subject,
    [Parameter(Mandatory)][string]$BodyText,
    [Parameter(Mandatory)][string]$BodyHtml,
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

  # Alternate Views
  $altText = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyText, $enc, "text/plain")
  $altText.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable

  $altHtml = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyHtml, $enc, "text/html")
  $altHtml.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable

  $msg.AlternateViews.Clear()
  $msg.AlternateViews.Add($altText)
  $msg.AlternateViews.Add($altHtml)

  $msg.Subject           = $Subject
  $msg.SubjectEncoding   = $enc
  $msg.Body              = $BodyText
  $msg.BodyEncoding      = $enc
  if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $enc }

  foreach ($p in $Attachments) {
    if ($p -and (Test-Path -LiteralPath $p)) {
      $att = New-Object System.Net.Mail.Attachment($p)
      if ($att.PSObject.Properties.Name -contains 'NameEncoding') {
        $att.NameEncoding = $enc
      }
      $msg.Attachments.Add($att) | Out-Null
    }
  }

  $client = New-Object System.Net.Mail.SmtpClient($SmtpHost, $SmtpPort)
  $client.EnableSsl = $true
  if ($User -and $Pass) {
    $client.Credentials = New-Object System.Net.NetworkCredential($User, $Pass)
  }

  try {
    Write-Log ("[MAIL] altViews count={0}, attach={1}" -f $msg.AlternateViews.Count, $msg.Attachments.Count)
    $client.Send($msg)
    if ($LogPath) { "[MAIL][OK] {0} {1} sent to {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Subject, ($ToList -join ", ") | Out-File -FilePath $LogPath -Append -Encoding UTF8 }
  } catch {
    $err = $_.Exception.Message
    if ($LogPath) { "[MAIL][ERROR] {0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $err | Out-File -FilePath $LogPath -Append -Encoding UTF8 }
    throw
  } finally {
    $msg.Dispose()
    $client.Dispose()
  }
}

# ---------- Send ----------
try {
  $sent = Send-ReportMail `
    -Subject   $subject `
    -BodyText  $plainBody `
    -BodyHtml  $BodyHtml `
    -ToList    $ToList `
    -From      $MAIL_FROM `
    -SmtpHost  $SMTP_HOST `
    -SmtpPort  $SMTP_PORT `
    -User      $SMTP_USER `
    -Pass      $SMTP_PASS `
    -Attachments $Attachments `
    -LogPath   $EmailLog

  Write-Log ("[DEBUG] saved html -> {0}" -f $debugHtmlPath)
  Write-Log "== DONE =="
  exit 0
} catch {
  Write-Log ("[FATAL] {0}" -f $_.Exception.Message)
  exit 1
}