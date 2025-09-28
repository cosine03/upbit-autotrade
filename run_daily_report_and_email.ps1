<#
run_daily_report_and_email.ps1 (DEDUP + HTML 본문에 표 미리보기)

- AM/PM 리포트 메일 발송
- 본문: merged/각 summary/dedup 상위 10행 미리보기 표 포함
- 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- 첨부: merged, breakout summary, box+line summary, dedup(있는 경우)

필수 .env 예 (Gmail):
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

# ===== 기본 경로/날짜 =====
$Root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$DATE  = Get-Date -Format "yyyy-MM-dd"

$DailyDir        = Join-Path $Root "logs\daily\${DATE}_$TagHalf"
$ReportsDir      = Join-Path $Root "logs\reports"
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

# ===== 로깅 =====
$EmailLog = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[{0}] {1}" -f $stamp, $msg
  $line  | Tee-Object -FilePath $EmailLog -Append
}
Write-Log "== RUN START == Root=$Root  Half=$TagHalf =="
Write-Log "DATE            : $DATE"
Write-Log "DailyDir        : $DailyDir"

# ===== .env 로딩 =====
$DotEnv = Join-Path $Root ".env"
function Load-DotEnv($path) {
  if (Test-Path -LiteralPath $path) {
    Get-Content $path | ForEach-Object {
      $line = $_.Trim()
      if (-not $line -or $line -match '^\s*#') { return }
      if ($line.Contains('#')) { $line = $line.Split('#')[0].Trim(); if (-not $line) { return } }
      $parts = $line.Split('=', 2)
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
    Write-Log "[WARN] .env not found at $path"
  }
}
Load-DotEnv $DotEnv

# ===== 주요 파일 경로 =====
$BtDirBreakout     = Join-Path $DailyDir "bt_breakout_only"
$BtDirBoxLine      = Join-Path $DailyDir "bt_boxin_linebreak"
$BreakoutSummary   = Join-Path $BtDirBreakout "bt_tv_events_stats_summary.csv"
$BoxLineSummary    = Join-Path $BtDirBoxLine  "bt_tv_events_stats_summary.csv"
$MergedSummary     = Join-Path $DailyDir ("bt_stats_summary_merged_{0}.csv" -f $TagHalf)
$MergedTradesDedup = Join-Path $DailyDir ("bt_trades_merged_dedup_{0}.csv" -f $TagHalf)

Write-Log "BreakoutSummary : $BreakoutSummary"
Write-Log "BoxLineSummary  : $BoxLineSummary"
Write-Log "MergedSummary   : $MergedSummary"
Write-Log "TradesDedup     : $MergedTradesDedup"

# ===== (옵션) 파이프라인 실행/병합 생성 =====
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

# ===== 메일 설정 =====
$SMTP_HOST = if ($env:SMTP_HOST) { $env:SMTP_HOST } else { 'smtp.gmail.com' }
$SMTP_PORT = if ($env:SMTP_PORT) { [int]$env:SMTP_PORT } else { 587 }
$SMTP_USER = $env:SMTP_USER
$SMTP_PASS = $env:SMTP_PASS
$MAIL_FROM = if ($env:MAIL_FROM) { $env:MAIL_FROM } else { $SMTP_USER }
$MAIL_TO   = $env:MAIL_TO

Write-Log "SMTP_HOST=$SMTP_HOST PORT=$SMTP_PORT USER=$SMTP_USER"
Write-Log "MAIL_FROM=$MAIL_FROM"
Write-Log "MAIL_TO=$MAIL_TO"

# ===== 표 미리보기 유틸 =====
Add-Type -AssemblyName System.Web  -ErrorAction SilentlyContinue

$PreviewRows = 10
$PreviewCols = 10

function Get-CsvPreview {
  param([string]$Path, [int]$MaxRows = 10, [int]$MaxCols = 10)
  if (-not (Test-Path -LiteralPath $Path)) { return $null }
  try {
    $rows = Import-Csv -LiteralPath $Path
    if (-not $rows) { return ,@() }          # 빈 CSV
    # 컬럼 제한
    $columns = $rows[0].PSObject.Properties.Name
    if ($columns.Count -gt $MaxCols) { $columns = $columns | Select-Object -First $MaxCols }
    # 행 제한
    $head = $rows | Select-Object -First $MaxRows
    # 선택 컬럼만 유지
    $trimmed = $head | Select-Object $columns
    return ,@($columns, $trimmed) # [0]=colnames, [1]=objects
  } catch {
    Write-Log "[CSV][ERROR] $($_.Exception.Message) on $Path"
    return $null
  }
}

function Convert-PreviewToHtmlTable {
  param([string[]]$Columns, [Object[]]$Objects, [string]$Caption)
  $encHtml = { param($s) if ($s -ne $null) { [System.Web.HttpUtility]::HtmlEncode([string]$s) } else { "" } }
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine('<table style="border-collapse:collapse;border:1px solid #ccc;font-size:13px;">')
  if ($Caption) {
    [void]$sb.AppendLine("<caption style='text-align:left;margin:8px 0;font-weight:bold;'>$([System.Web.HttpUtility]::HtmlEncode($Caption))</caption>")
  }
  # 헤더
  [void]$sb.AppendLine('<thead><tr>')
  foreach ($c in $Columns) {
    [void]$sb.AppendLine("<th style='border:1px solid #ccc;background:#f7f7f7;padding:4px 6px;'>$([System.Web.HttpUtility]::HtmlEncode($c))</th>")
  }
  [void]$sb.AppendLine('</tr></thead>')
  # 바디
  [void]$sb.AppendLine('<tbody>')
  foreach ($o in $Objects) {
    [void]$sb.AppendLine('<tr>')
    foreach ($c in $Columns) {
      $val = if ($o.PSObject.Properties.Match($c)) { $o.$c } else { "" }
      [void]$sb.AppendLine("<td style='border:1px solid #eee;padding:4px 6px;'>$(&$encHtml $val)</td>")
    }
    [void]$sb.AppendLine('</tr>')
  }
  [void]$sb.AppendLine('</tbody></table>')
  return $sb.ToString()
}

# ===== 각 CSV 미리보기 생성 =====
$mergedPrev  = Get-CsvPreview -Path $MergedSummary     -MaxRows $PreviewRows -MaxCols $PreviewCols
$breakPrev   = Get-CsvPreview -Path $BreakoutSummary   -MaxRows $PreviewRows -MaxCols $PreviewCols
$boxPrev     = Get-CsvPreview -Path $BoxLineSummary    -MaxRows $PreviewRows -MaxCols $PreviewCols
$dedupPrev   = Get-CsvPreview -Path $MergedTradesDedup -MaxRows $PreviewRows -MaxCols $PreviewCols

$htmlTables = @()
if ($mergedPrev) { $htmlTables += (Convert-PreviewToHtmlTable -Columns $mergedPrev[0] -Objects $mergedPrev[1] -Caption "Merged Summary (Top $PreviewRows x $PreviewCols)") }
if ($breakPrev)  { $htmlTables += (Convert-PreviewToHtmlTable -Columns $breakPrev[0]  -Objects $breakPrev[1]  -Caption "Breakout Summary (Top $PreviewRows x $PreviewCols)") }
if ($boxPrev)    { $htmlTables += (Convert-PreviewToHtmlTable -Columns $boxPrev[0]    -Objects $boxPrev[1]    -Caption "Box+Line Summary (Top $PreviewRows x $PreviewCols)") }
if ($dedupPrev)  { $htmlTables += (Convert-PreviewToHtmlTable -Columns $dedupPrev[0]  -Objects $dedupPrev[1]  -Caption "Trades Dedup (Top $PreviewRows x $PreviewCols)") }

# ===== 메일 본문(HTML) =====
$subject = "[Autotrade] Daily Report $DATE $TagHalf"
$body = @"
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body { font-family: Arial, sans-serif; font-size: 14px; color:#222; }
  h2   { color: #2a4b8d; }
  .section { margin: 18px 0; }
  ul   { margin: 0 0 12px 18px; padding: 0; }
</style>
</head>
<body>
  <h2>📊 Autotrade Daily Report ($DATE $TagHalf)</h2>
  <div class="section">
    <p>자동 생성된 리포트입니다. 주요 CSV 파일은 첨부로도 포함되어 있습니다.</p>
    <ul>
      <li><b>요약:</b> $(Split-Path $MergedSummary -Leaf)</li>
      <li><b>Breakout:</b> $(Split-Path $BreakoutSummary -Leaf)</li>
      <li><b>Box+Line:</b> $(Split-Path $BoxLineSummary -Leaf)</li>
      <li><b>Dedup Trades:</b> $(Split-Path $MergedTradesDedup -Leaf)</li>
    </ul>
    <p>로그 파일: $(Resolve-Path $EmailLog)</p>
  </div>

  <div class="section">
    $(($htmlTables -join "<br/>"))
  </div>
</body>
</html>
"@

# ===== 메일 발송 함수 (UTF-8 강제) =====
$enc = [System.Text.Encoding]::UTF8
$msg = New-Object System.Net.Mail.MailMessage
$msg.From = $From
foreach ($to in $ToList) { $msg.To.Add($to) }

$msg.Subject         = $Subject
$msg.SubjectEncoding = $enc

# ★ HTML 본문을 AlternateView로 UTF-8 + Base64 강제
$msg.IsBodyHtml   = $true
$msg.Body         = ""             # (본문은 AlternateView로만 보냄)
$msg.BodyEncoding = $enc
if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $enc }

# AlternateView 생성 (text/html, UTF-8)
$altHtml = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($Body, $enc, "text/html")
$altHtml.TransferEncoding = [System.Net.Mime.TransferEncoding]::Base64
# (선택) ContentType.CharSet 보강
$altHtml.ContentType.CharSet = "utf-8"

$msg.AlternateViews.Clear()
$msg.AlternateViews.Add($altHtml)

# 첨부 파일 (파일명 인코딩)
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
$client.Send($msg)

# ===== 수신자/첨부 구성 후 발송 =====
$ToList = $MAIL_TO.Split(',;') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if (-not $ToList) { Write-Log "[MAIL][ERROR] MAIL_TO empty"; throw "MAIL_TO empty" }

$Attachments = @()
foreach ($p in @($MergedSummary,$BreakoutSummary,$BoxLineSummary,$MergedTradesDedup)) {
  if (Test-Path $p) { $Attachments += (Resolve-Path $p).Path }
  else { Write-Log "[ATTACH][WARN] not found -> $p" }
}

try {
  Send-ReportMail -Subject $subject -Body $body -ToList $ToList -From $MAIL_FROM `
    -SmtpHost $SMTP_HOST -SmtpPort $SMTP_PORT -User $SMTP_USER -Pass $SMTP_PASS `
    -Attachments $Attachments -LogPath $EmailLog
  Write-Log "== DONE =="
  exit 0
} catch {
  Write-Log "[FATAL] $($_.Exception.Message)"
  exit 1
}