<# 
run_daily_report_and_email.ps1
- AM/PM별 일일 리포트 생성(옵션) + 메일 발송(UTF-8, HTML)
- 메일 전송 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- 첨부: 
  1) bt_stats_summary_merged_{AM|PM}.csv
  2) bt_breakout_only\bt_tv_events_stats_summary.csv
  3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv

.env 예:
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password_here        # 앱 비밀번호
MAIL_FROM=you@gmail.com
MAIL_TO=a@ex.com,b@ex.com
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

# ---------- Logging ----------
$EmailLog = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[{0}] {1}" -f $stamp, $msg
  $line | Tee-Object -FilePath $EmailLog -Append
}
Write-Log "== RUN START == Root=$Root  Half=$TagHalf =="

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
      $k = $parts[0].Trim(); $v = $parts[1].Trim()
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
$TradesMergedCsv  = Join-Path $DailyDir ("bt_trades_merged_dedup_{0}.csv" -f $TagHalf) # 선택: dedup 결과 저장

Write-Log "DailyDir       : $DailyDir"
Write-Log "MergedSummary  : $MergedSummary"

# ---------- Optional pipeline (merge + dedup) ----------
function Merge-And-Dedup {
  param(
    [string]$BreakoutSummaryCsv,
    [string]$BoxLineSummaryCsv,
    [string]$MergedOutCsv
  )
  if ((Test-Path $BreakoutSummaryCsv) -and (Test-Path $BoxLineSummaryCsv)) {
    $b = Import-Csv $BreakoutSummaryCsv
    $l = Import-Csv $BoxLineSummaryCsv
    $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
    $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
    ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedOutCsv
    Write-Log "merged summary saved -> $MergedOutCsv"
  } else {
    Write-Log "[PIPE][WARN] summary files missing; skip merge."
  }
}

# (필요 시) trades dedup 유틸: 동일 symbol/event/expiry/entry_ts/side 기준 중복 제거
function Dedup-Trades {
  param(
    [string[]]$TradeCsvPaths,
    [string]$OutCsv
  )
  $all = @()
  foreach ($p in $TradeCsvPaths) {
    if (Test-Path $p) { $all += (Import-Csv $p) }
  }
  if (-not $all -or $all.Count -eq 0) { return }
  $keys = @{}
  $dedup = New-Object System.Collections.Generic.List[object]
  foreach ($row in $all) {
    $k = "{0}|{1}|{2}|{3}|{4}" -f $row.symbol,$row.event,$row.expiry_h,$row.entry_ts,$row.side
    if (-not $keys.ContainsKey($k)) { $keys[$k] = $true; $dedup.Add($row) }
  }
  $dedup | Export-Csv -NoTypeInformation -Encoding UTF8 $OutCsv
  Write-Log "dedup trades saved -> $OutCsv (in=${all.Count}, out=${dedup.Count})"
}

if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }

    # (여기: 신호 분리/백테스트 실행이 필요하면 추가 호출)
    # 예시로 summary 병합만 수행
    Merge-And-Dedup -BreakoutSummaryCsv $BreakoutSummary -BoxLineSummaryCsv $BoxLineSummary -MergedOutCsv $MergedSummary

    # (선택) 트레이드 dedup: bt_*_trades.csv 찾기
    $tradeCsvs = @()
    $btBreakTrades = Join-Path $BtDirBreakout "bt_tv_events_trades.csv"
    $btBoxTrades   = Join-Path $BtDirBoxLine  "bt_tv_events_trades.csv"
    if (Test-Path $btBreakTrades) { $tradeCsvs += $btBreakTrades }
    if (Test-Path $btBoxTrades)   { $tradeCsvs += $btBoxTrades }
    if ($tradeCsvs.Count -gt 0) {
      Dedup-Trades -TradeCsvPaths $tradeCsvs -OutCsv $TradesMergedCsv
    }

    Write-Log "[PIPE] done"
  } catch {
    Write-Log "[PIPE][ERROR] $($_.Exception.Message)"
  }
} else {
  if ((-not (Test-Path $MergedSummary)) -and (Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
    Merge-And-Dedup -BreakoutSummaryCsv $BreakoutSummary -BoxLineSummaryCsv $BoxLineSummary -MergedOutCsv $MergedSummary
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

# ---------- Recipients & Attachments ----------
# 수신자 배열 정리(콤마/세미콜론 허용)
$ToList = @()
if ($MAIL_TO) {
  $MAIL_TO.Split(',;') | ForEach-Object { $addr = $_.Trim(); if ($addr) { $ToList += $addr } }
}
if (-not $ToList -or $ToList.Count -eq 0) { Write-Log "[MAIL][ERROR] MAIL_TO empty."; throw "MAIL_TO empty" }

$Attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path $p) { $Attachments += (Resolve-Path $p).Path } else { Write-Log "[ATTACH][WARN] not found -> $p" }
}
if (Test-Path $TradesMergedCsv) { $Attachments += (Resolve-Path $TradesMergedCsv).Path }

# ---------- HTML body (UTF-8, 테이블 포함) ----------
# 테이블 미리뷰: merged summary 상위 10행
function Get-TopHtmlTable {
  param([string]$CsvPath, [int]$Top = 10)
  if (-not (Test-Path $CsvPath)) { return "<p>요약 파일이 없습니다: $([System.Web.HttpUtility]::HtmlEncode($CsvPath))</p>" }
  $rows = Import-Csv $CsvPath | Select-Object -First $Top
  if (-not $rows) { return "<p>표시할 데이터가 없습니다.</p>" }

  $cols = $rows[0].PSObject.Properties.Name
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine('<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;font-family:Segoe UI,Arial,sans-serif;font-size:12px;">')
  [void]$sb.AppendLine('<thead><tr>')
  foreach ($c in $cols) { [void]$sb.AppendLine("<th style='background:#f2f2f2;text-align:left;'>$([System.Web.HttpUtility]::HtmlEncode($c))</th>") }
  [void]$sb.AppendLine('</tr></thead><tbody>')
  foreach ($r in $rows) {
    [void]$sb.AppendLine('<tr>')
    foreach ($c in $cols) {
      $val = $r.$c
      [void]$sb.AppendLine("<td>$([System.Web.HttpUtility]::HtmlEncode([string]$val))</td>")
    }
    [void]$sb.AppendLine('</tr>')
  }
  [void]$sb.AppendLine('</tbody></table>')
  $sb.ToString()
}

$Subject = "[Autotrade] Daily Report $DATE $TagHalf"
$TopTableHtml = Get-TopHtmlTable -CsvPath $MergedSummary -Top 10
$BodyHtml = @"
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>$([System.Web.HttpUtility]::HtmlEncode($Subject))</title>
</head>
<body style="font-family:Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.5;">
  <p>자동 생성 리포트 <strong>($DATE $TagHalf)</strong></p>
  <p>첨부 파일:</p>
  <ul>
    <li>$(Split-Path $MergedSummary -Leaf)</li>
    <li>$(Split-Path $BreakoutSummary -Leaf)</li>
    <li>$(Split-Path $BoxLineSummary -Leaf)</li>
    $(if (Test-Path $TradesMergedCsv) { "<li>$(Split-Path $TradesMergedCsv -Leaf)</li>" })
  </ul>
  <h3>요약 미리보기 (상위 10행)</h3>
  $TopTableHtml
  <p style="margin-top:16px;">로그: $(Resolve-Path $EmailLog)</p>
</body>
</html>
"@

# ---------- Mail sender (System.Net.Mail; UTF-8 + HTML) ----------
function Send-ReportMail {
  param(
    [Parameter(Mandatory)][string]$Subject,
    [Parameter(Mandatory)][string]$BodyHtml,
    [Parameter(Mandatory)][string[]]$ToList,
    [Parameter(Mandatory)][string]$From,
    [Parameter(Mandatory)][string]$SmtpHost,
    [int]$SmtpPort = 587,
    [string]$User = $null,
    [string]$Pass = $null,
    [string[]]$Attachments = @(),
    [string]$LogPath = $null
  )

  $enc = [System.Text.Encoding]::UTF8

  $msg = New-Object System.Net.Mail.MailMessage
  $msg.From = $From
  foreach ($to in $ToList) { [void]$msg.To.Add($to) }

  $msg.Subject          = $Subject
  $msg.SubjectEncoding  = $enc
  $msg.IsBodyHtml       = $true

  # HTML AlternateView (UTF-8 + Quoted-Printable)
  $alt = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyHtml, $enc, "text/html")
  $alt.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
  $msg.AlternateViews.Add($alt)

  # 호환용 Body에도 HTML 세팅 + 인코딩 강제
  $msg.Body         = $BodyHtml
  $msg.BodyEncoding = $enc
  if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $enc }

  foreach ($p in $Attachments) {
    if ($p -and (Test-Path -LiteralPath $p)) {
      $att = New-Object System.Net.Mail.Attachment($p)
      if ($att.PSObject.Properties.Name -contains 'NameEncoding') { $att.NameEncoding = $enc }
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
    $msg.Dispose()
    $client.Dispose()
  }
}

# ---------- Send ----------
try {
  Send-ReportMail `
    -Subject  $Subject `
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