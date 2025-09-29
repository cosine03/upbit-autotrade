<# 
run_daily_report_and_email.ps1  (Clean UTF-8 / 3 tables / percent formatting / safe logging)

- AM/PM별 일일 리포트 생성(옵션) + 메일 발송
- 메일 전송 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log (매 실행 시 초기화)
- 첨부: 
  1) bt_stats_summary_merged_{AM|PM}.csv
  2) bt_breakout_only\bt_tv_events_stats_summary.csv
  3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv

.env (예: Gmail)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password_here   # 앱 비밀번호
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

# ---------- Paths & Date ----------
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$DATE = Get-Date -Format "yyyy-MM-dd"

$DailyDir    = Join-Path $Root "logs\daily\${DATE}_$TagHalf"
$ReportsDir  = Join-Path $Root "logs\reports"
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

$EmailLog    = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
# 매 실행 시 같은 파일 초기화(섞임 방지)
if (Test-Path $EmailLog) { Clear-Content -LiteralPath $EmailLog -ErrorAction SilentlyContinue }

# ---------- Logger (UTF-8 강제) ----------
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[{0}] {1}" -f $stamp, $msg
  Write-Host $line
  Add-Content -LiteralPath $EmailLog -Value $line -Encoding UTF8
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
  } else {
    Write-Log "[WARN] .env not found: $path"
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

Write-Log "DailyDir=$DailyDir"
Write-Log "[CHECK] exists merged=$([int](Test-Path $MergedSummary)) breakout=$([int](Test-Path $BreakoutSummary)) boxline=$([int](Test-Path $BoxLineSummary))"

# ---------- Optional pipeline ----------
if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }

    # (필요 시 여기에: 신호 분리/백테스트 실행 커맨드 호출)
    # 예: python .\backtest_tv_events_mp.py "$SignalsBreakout" ... --outdir "$BtDirBreakout"
    #     python .\backtest_tv_events_mp.py "$SignalsBoxLine"    ... --outdir "$BtDirBoxLine"

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

# ---------- Helpers: HTML tables ----------
Add-Type -AssemblyName System.Web | Out-Null

function Format-Percent2([double]$x) {
  if ($null -eq $x) { return "" }
  # 들어오는 값이 0.615 형태이므로 P2로 표시
  return ("{0:P2}" -f $x)
}

function Get-TopHtmlTable {
  param(
    [Parameter(Mandatory)][string]$CsvPath,
    [Parameter(Mandatory)][string]$Title,
    [int]$Top = 10
  )
  if (-not (Test-Path $CsvPath)) {
    Write-Log "[HTML] $Title file missing -> $CsvPath"
    return ""
  }
  try {
    $d = Import-Csv $CsvPath
    if (-not $d -or $d.Count -eq 0) {
      Write-Log "[HTML] $Title empty html -> $CsvPath"
      return ""
    }

    # 필요한 컬럼만 추출 & 상위 Top
    $rows = $d | Select-Object `
      strategy, event, expiry_h, trades, win_rate, avg_net, median_net, total_net `
      | Select-Object -First $Top

    # 숫자 포맷팅
    foreach ($r in $rows) {
      if ($r.win_rate  -ne $null) { $r.win_rate  = Format-Percent2([double]$r.win_rate) }
      if ($r.avg_net   -ne $null) { $r.avg_net   = Format-Percent2([double]$r.avg_net) }
      if ($r.median_net-ne $null) { $r.median_net= Format-Percent2([double]$r.median_net) }
      if ($r.total_net -ne $null) { $r.total_net = Format-Percent2([double]$r.total_net) }
      if ($r.trades    -ne $null) { $r.trades    = ("{0:N0}" -f [double]$r.trades) }
      if ($r.expiry_h  -ne $null) { $r.expiry_h  = ("{0:N1}" -f [double]$r.expiry_h) }
    }

    $cols = @('strategy','event','expiry_h','trades','win_rate','avg_net','median_net','total_net')
    $sb = New-Object System.Text.StringBuilder

    [void]$sb.AppendLine("<h3 style='margin:16px 0 6px 0;'>$([System.Web.HttpUtility]::HtmlEncode($Title))</h3>")
    [void]$sb.AppendLine("<table style='border-collapse:collapse;width:100%;font-family:Segoe UI,Arial,sans-serif;font-size:12.5px;'>")
    # head
    [void]$sb.AppendLine("<thead><tr>")
    foreach ($c in $cols) {
      [void]$sb.AppendLine("<th style='border:1px solid #ddd;background:#f6f6f6;text-align:left;padding:6px;'>$([System.Web.HttpUtility]::HtmlEncode($c))</th>")
    }
    [void]$sb.AppendLine("</tr></thead>")

    # body
    [void]$sb.AppendLine("<tbody>")
    foreach ($r in $rows) {
      [void]$sb.AppendLine("<tr>")
      foreach ($c in $cols) {
        $val = $r.$c
        [void]$sb.AppendLine("<td style='border:1px solid #ddd;padding:6px;'>$([System.Web.HttpUtility]::HtmlEncode([string]$val))</td>")
      }
      [void]$sb.AppendLine("</tr>")
    }
    [void]$sb.AppendLine("</tbody></table>")

    $html = $sb.ToString()
    Write-Log "[HTML] $Title ok length=$($html.Length)"
    return $html
  } catch {
    Write-Log "[HTML][ERROR] $Title -> $($_.Exception.Message)"
    return ""
  }
}

# ---------- Build 3 sections (merged / breakout / boxline) ----------
$MergedHtml  = if (Test-Path $MergedSummary)  { Get-TopHtmlTable -CsvPath $MergedSummary -Title "Merged Summary (Top 10)" } else { "" }
$BreakHtml   = if (Test-Path $BreakoutSummary){ Get-TopHtmlTable -CsvPath $BreakoutSummary -Title "Breakout Only (Top 10)" } else { "" }
$BoxLineHtml = if (Test-Path $BoxLineSummary) { Get-TopHtmlTable -CsvPath $BoxLineSummary -Title "Box-in Line-break (Top 10)" } else { "" }

# ---------- Subject / Body ----------
$Subject = "[Autotrade] Daily Report $DATE $TagHalf"

$BodyHtml = @"
<!doctype html>
<html>
<head>
<meta charset="UTF-8" />
<title>$Subject</title>
</head>
<body style="margin:0;padding:0;font-family:Segoe UI,Arial,sans-serif;background:#fafafa;">
  <div style="max-width:860px;margin:0 auto;padding:16px 14px 24px 14px;background:white;border:1px solid #eee;">
    <h2 style="margin:6px 0 14px 0;">$Subject</h2>
    <div style="font-size:12.5px;color:#555;margin-bottom:8px;">Auto-generated report. See details below.</div>
    $MergedHtml
    $BreakHtml
    $BoxLineHtml
  </div>
</body>
</html>
"@

# Text fallback (간단 영어)
$BodyText = @"
$Subject

This email includes three summaries (Top 10 rows each):
- Merged Summary
- Breakout Only
- Box-in Line-break

Attachments:
- $(Split-Path $MergedSummary -Leaf)
- $(Split-Path $BreakoutSummary -Leaf)
- $(Split-Path $BoxLineSummary -Leaf)

Log: $EmailLog
"@

# ---------- Attachments ----------
$Attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path $p) { $Attachments += (Resolve-Path $p).Path } else { Write-Log "[ATTACH][WARN] not found -> $p" }
}

# ---------- Save debug HTML ----------
$debugHtmlPath = Join-Path $ReportsDir ("email_{0}_{1}.html" -f $DATE, $TagHalf)
$BodyHtml | Out-File -FilePath $debugHtmlPath -Encoding UTF8
Write-Log "[DEBUG] saved html -> $debugHtmlPath"

# ---------- Send email (System.Net.Mail; UTF-8 & AltView) ----------
# 수신자 배열
$ToList = @()
if ($MAIL_TO) {
  $MAIL_TO.Split(',;') | ForEach-Object { $addr = $_.Trim(); if ($addr) { $ToList += $addr } }
}
if (-not $ToList -or $ToList.Count -eq 0) { Write-Log "[MAIL][ERROR] MAIL_TO empty."; throw "MAIL_TO empty" }

$enc = [System.Text.Encoding]::UTF8

$msg = New-Object System.Net.Mail.MailMessage
$msg.From = $MAIL_FROM
foreach ($to in $ToList) { [void]$msg.To.Add($to) }
$msg.Subject          = $Subject
$msg.SubjectEncoding  = $enc
$msg.IsBodyHtml       = $false
$msg.Body             = $BodyText
$msg.BodyEncoding     = $enc
if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $enc }

# HTML AlternateView (UTF-8 / Quoted-Printable)
$altHtml = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyHtml, $enc, "text/html")
$altHtml.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
$msg.AlternateViews.Add($altHtml) | Out-Null

# Attach
foreach ($p in $Attachments) {
  $att = New-Object System.Net.Mail.Attachment($p)
  if ($att.PSObject.Properties.Name -contains 'NameEncoding') { $att.NameEncoding = $enc }
  $msg.Attachments.Add($att) | Out-Null
}

# Send
$client = New-Object System.Net.Mail.SmtpClient($SMTP_HOST, $SMTP_PORT)
$client.EnableSsl = $true
if ($SMTP_USER -and $SMTP_PASS) {
  $client.Credentials = New-Object System.Net.NetworkCredential($SMTP_USER, $SMTP_PASS)
}

try {
  Write-Log "[MAIL] altViews count=$($msg.AlternateViews.Count), attach=$($msg.Attachments.Count)"
  $client.Send($msg)
  Write-Log "[MAIL][OK] $Subject sent to $($ToList -join ', ')"
  Write-Log "== DONE =="
  exit 0
} catch {
  Write-Log "[FATAL] $($_.Exception.Message)"
  exit 1
} finally {
  $msg.Dispose()
  $client.Dispose()
}