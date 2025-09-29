<# 
run_daily_report_and_email.ps1
- AM/PM별 리포트(선택: 파이프라인 실행) + 메일 발송
- 로그: .\logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- 첨부: 
  1) logs/daily/{DATE}_{AM|PM}/bt_stats_summary_merged_{AM|PM}.csv
  2) .../bt_breakout_only/bt_tv_events_stats_summary.csv
  3) .../bt_boxin_linebreak/bt_tv_events_stats_summary.csv

.env 예(Gmail):
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password_or_token
MAIL_FROM=you@gmail.com
MAIL_TO=a@example.com,b@example.com
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
$DATE       = Get-Date -Format "yyyy-MM-dd"
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
function Load-DotEnv($path) {
  if (-not (Test-Path -LiteralPath $path)) { Write-Log "[WARN] .env not found at $path"; return }
  Get-Content $path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line) { return }
    if ($line -match '^\s*#') { return }
    if ($line.Contains('#')) { $line = $line.Split('#')[0].Trim(); if (-not $line) { return } }
    $parts = $line.Split('=',2); if ($parts.Count -ne 2) { return }
    $k = $parts[0].Trim(); $v = $parts[1].Trim()
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

Write-Log "DailyDir=$DailyDir"
Write-Log "[CHECK] exists merged=$([int](Test-Path $MergedSummary)) breakout=$([int](Test-Path $BreakoutSummary)) boxline=$([int](Test-Path $BoxLineSummary))"

# ---------- Optional: run pipeline before email ----------
if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }

    # (필요 시 여기에 신호 분리/백테스트 실행 커맨드 삽입)

    # 요약 병합(있을 때만)
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

# ---------- Small helpers ----------
Add-Type -AssemblyName System.Web | Out-Null

function Format-Percent2($x) {
  if ($null -eq $x -or $x -eq '') { return '' }
  try {
    $v = [double]$x
    return ("{0:N2}%" -f ($v * 100.0))
  } catch { return [string]$x }
}

function Get-TopHtmlTable {
  param(
    [Parameter(Mandatory)][string]$CsvPath,
    [int]$Top = 10
  )
  if (-not (Test-Path -LiteralPath $CsvPath)) { return "" }
  $rows = Import-Csv -Path $CsvPath | Select-Object -First $Top
  if (-not $rows -or $rows.Count -eq 0) { return "" }
  $cols = $rows[0].PSObject.Properties.Name

  # 숫자 컬럼 포맷(who: percent 2 dec)
  foreach ($r in $rows) {
    foreach ($c in @('win_rate','avg_net','median_net','total_net')) {
      if ($cols -contains $c) { $r.$c = Format-Percent2 $r.$c }
    }
  }

  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine("<table style='border-collapse:collapse;width:100%;margin:8px 0;font-size:13px;'>")
  [void]$sb.AppendLine("<thead><tr>")
  foreach ($c in $cols) {
    $th = [System.Web.HttpUtility]::HtmlEncode($c)
    [void]$sb.AppendLine("<th style='background:#f2f2f2;text-align:left;border:1px solid #ddd;padding:6px;'>$th</th>")
  }
  [void]$sb.AppendLine("</tr></thead><tbody>")
  foreach ($r in $rows) {
    [void]$sb.AppendLine("<tr>")
    foreach ($c in $cols) {
      $val = [System.Web.HttpUtility]::HtmlEncode([string]$r.$c)
      [void]$sb.AppendLine("<td style='border:1px solid #ddd;padding:6px;'>$val</td>")
    }
    [void]$sb.AppendLine("</tr>")
  }
  [void]$sb.AppendLine("</tbody></table>")
  return $sb.ToString()
}

function Build-TableHtmlOrNote {
  param([string]$Path, [string]$Label, [int]$Top = 10)

  if (-not (Test-Path -LiteralPath $Path)) {
    Write-Log "[HTML] $($Label): file missing -> $Path"
    return "<p style='color:#999'>(no data for $($Label))</p>"
  }
  try {
    $ht = Get-TopHtmlTable -CsvPath $Path -Top $Top
    if ([string]::IsNullOrWhiteSpace($ht)) {
      Write-Log "[HTML] $($Label): empty html -> $Path"
      return "<p style='color:#999'>(no data for $($Label))</p>"
    }
    Write-Log "[HTML] $($Label): ok length=$($ht.Length)"
    return $ht
  } catch {
    Write-Log "[HTML][ERROR] $($Label): $($_.Exception.Message)"
    return "<p style='color:#c00'>(error rendering $($Label))</p>"
  }
}

# ---------- Compose subject/body ----------
$Subject = "[Autotrade] Daily Report $DATE $TagHalf"

$MergedHtml  = Build-TableHtmlOrNote -Path $MergedSummary  -Label "merged"  -Top 10
$BreakHtml   = Build-TableHtmlOrNote -Path $BreakoutSummary -Label "breakout" -Top 10
$BoxLineHtml = Build-TableHtmlOrNote -Path $BoxLineSummary  -Label "boxline"  -Top 10

# text/plain (카카오 미리보기용 간결 버전)
$BodyText = @"
Autotrade Daily Report ($DATE $TagHalf)

Attachments: 3
- $(Split-Path $MergedSummary -Leaf)
- $(Split-Path $BreakoutSummary -Leaf)
- $(Split-Path $BoxLineSummary -Leaf)

Log: $(Resolve-Path $EmailLog)
"@.Trim()

# HTML 본문
$BodyHtml = @"
<!doctype html>
<html>
<head>
  <meta charset="UTF-8"/>
  <style>
    body{font-family:Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.4}
    h2{margin:8px 0 4px 0}
    h3{margin:14px 0 6px 0}
    .muted{color:#666}
    .wrap{max-width:980px}
  </style>
</head>
<body>
<div class="wrap">
  <h2>Autotrade Daily Report ($DATE $TagHalf)</h2>
  <p class="muted">Attachments: 3</p>

  <h3>Merged Summary (Top 10)</h3>
  $MergedHtml

  <h3>Breakout Summary (Top 10)</h3>
  $BreakHtml

  <h3>Box-Line Summary (Top 10)</h3>
  $BoxLineHtml

  <p class="muted">Log: $(Resolve-Path $EmailLog)</p>
</div>
</body>
</html>
"@.Trim()

# ---------- Attachments ----------
$Attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path $p) { $Attachments += (Resolve-Path $p).Path } else { Write-Log "[ATTACH][WARN] not found -> $p" }
}

# 디버그용 HTML 저장 경로 정의
$debugHtmlPath = Join-Path $ReportsDir ("email_{0}_{1}.html" -f $DATE, $TagHalf)

# (선택) 본문 HTML이 $BodyHtml 변수라면 파일로 저장
$BodyHtml | Out-File -FilePath $debugHtmlPath -Encoding UTF8

# ---------- Send (System.Net.Mail with AlternateViews) ----------
$enc = [System.Text.Encoding]::UTF8
$msg = New-Object System.Net.Mail.MailMessage
if (-not $MAIL_FROM) { throw "MAIL_FROM empty" }
$msg.From = $MAIL_FROM

# recipients
$ToList = @()
if ($MAIL_TO) {
  $MAIL_TO.Split(',;') | ForEach-Object { $addr = $_.Trim(); if ($addr) { $ToList += $addr } }
}
if ($ToList.Count -eq 0) { throw "MAIL_TO empty" }
$ToList | ForEach-Object { [void]$msg.To.Add($_) }

$msg.Subject          = $Subject
$msg.SubjectEncoding  = $enc
$msg.IsBodyHtml       = $false   # 본문은 AlternateViews 로 제공

# Alternate views: plain -> html
$altPlain = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyText, $enc, "text/plain")
$altPlain.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
$altHtml  = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyHtml, $enc, "text/html")
$altHtml.TransferEncoding  = [System.Net.Mime.TransferEncoding]::QuotedPrintable
$msg.AlternateViews.Add($altPlain)
$msg.AlternateViews.Add($altHtml)

# Attach
foreach ($p in $Attachments) {
  $att = New-Object System.Net.Mail.Attachment($p)
  if ($att.PSObject.Properties.Name -contains 'NameEncoding') { $att.NameEncoding = $enc }
  $msg.Attachments.Add($att) | Out-Null
}

$smtp = New-Object System.Net.Mail.SmtpClient($SMTP_HOST, $SMTP_PORT)
$smtp.EnableSsl = $true
if ($SMTP_USER -and $SMTP_PASS) {
  $cred = New-Object System.Net.NetworkCredential($SMTP_USER, $SMTP_PASS)
  $smtp.Credentials = $cred
}

# 전송
try {
  Write-Log ("[MAIL] altViews count={0}, attach={1}" -f $msg.AlternateViews.Count, $msg.Attachments.Count)
  # 디버그용: 보낸 HTML 저장
  $debugHtmlPath = Join-Path $ReportsDir ("debug_body_{0}_{1}.html" -f $DATE, $TagHalf)
  $BodyHtml | Out-File -FilePath $debugHtmlPath -Encoding UTF8
  Write-Log "[DEBUG] saved html -> $debugHtmlPath"

  $smtp.Send($msg)
  Write-Log "== DONE =="
  exit 0
} catch {
  Write-Log "[FATAL] $($_.Exception.Message)"
  exit 1
} finally {
  $msg.Dispose()
  $smtp.Dispose()
}