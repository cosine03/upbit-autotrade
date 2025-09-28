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
function Write-Log([string]$msg){
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "[{0}] {1}" -f $stamp,$msg | Tee-Object -FilePath $EmailLog -Append
}
Write-Log "== RUN START == Root=$Root Half=$TagHalf =="

# ---------- .env loader ----------
function Load-DotEnv([string]$path){
  if (Test-Path -LiteralPath $path){
    Get-Content $path | ForEach-Object{
      $line = $_.Trim()
      if (-not $line -or $line -match '^\s*#') { return }
      if ($line.Contains('#')) { $line = $line.Split('#')[0].Trim(); if (-not $line){ return } }
      $kv = $line.Split('=',2)
      if ($kv.Count -ne 2){ return }
      $k = $kv[0].Trim(); $v = $kv[1].Trim()
      if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))){
        $v = $v.Substring(1,$v.Length-2)
      }
      if ($k){ Set-Item -Path ("Env:{0}" -f $k) -Value $v }
    }
    Write-Log ".env loaded."
  } else {
    Write-Log "[WARN] .env not found: $path"
  }
}
Load-DotEnv (Join-Path $Root ".env")

# ---------- Important files ----------
$BtDirBreakout   = Join-Path $DailyDir "bt_breakout_only"
$BtDirBoxLine    = Join-Path $DailyDir "bt_boxin_linebreak"
$BreakoutSummary = Join-Path $BtDirBreakout "bt_tv_events_stats_summary.csv"
$BoxLineSummary  = Join-Path $BtDirBoxLine  "bt_tv_events_stats_summary.csv"
$MergedSummary   = Join-Path $DailyDir ("bt_stats_summary_merged_{0}.csv" -f $TagHalf)

# ---------- (Optional) Merge summaries ----------
function Merge-Summaries {
  param([string]$BreakoutCsv,[string]$BoxLineCsv,[string]$OutCsv)
  if ((Test-Path $BreakoutCsv) -and (Test-Path $BoxLineCsv)){
    $b = Import-Csv $BreakoutCsv
    $l = Import-Csv $BoxLineCsv
    $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
    $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
    ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $OutCsv
    Write-Log "merged summary saved -> $OutCsv"
  } else {
    Write-Log "[PIPE][WARN] missing summary -> skip merge"
  }
}

if ($RunPipeline){
  if (-not (Test-Path $DailyDir)){ New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }
  Merge-Summaries -BreakoutCsv $BreakoutSummary -BoxLineCsv $BoxLineSummary -OutCsv $MergedSummary
} else {
  if ((-not (Test-Path $MergedSummary)) -and (Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)){
    Merge-Summaries -BreakoutCsv $BreakoutSummary -BoxLineCsv $BoxLineSummary -OutCsv $MergedSummary
  }
}

# ---------- SMTP config ----------
$SMTP_HOST = $env:SMTP_HOST; if (-not $SMTP_HOST){ $SMTP_HOST = 'smtp.gmail.com' }
$SMTP_PORT = if ($env:SMTP_PORT) { [int]$env:SMTP_PORT } else { 587 }
$SMTP_USER = $env:SMTP_USER
$SMTP_PASS = $env:SMTP_PASS
$MAIL_FROM = if ($env:MAIL_FROM) { $env:MAIL_FROM } else { $SMTP_USER }
$MAIL_TO   = $env:MAIL_TO

Write-Log "SMTP_HOST=$SMTP_HOST PORT=$SMTP_PORT USER=$SMTP_USER"
Write-Log "MAIL_FROM=$MAIL_FROM"
Write-Log "MAIL_TO=$MAIL_TO"

# ---------- Recipients / Attachments ----------
$ToList = @()
if ($MAIL_TO){ $MAIL_TO.Split(',;') | % { $a=$_.Trim(); if($a){ $ToList+=$a } } }
if (-not $ToList){ throw "MAIL_TO empty" }

$Attachments = @()
foreach($p in @($MergedSummary,$BreakoutSummary,$BoxLineSummary)){
  if (Test-Path $p){ $Attachments += (Resolve-Path $p).Path } else { Write-Log "[ATTACH][WARN] missing -> $p" }
}

# ---------- HTML helpers ----------
Add-Type -AssemblyName System.Web | Out-Null
function Get-TopHtmlTable {
  param([string]$CsvPath,[int]$Top=10)
  if (-not (Test-Path $CsvPath)) { return "<p>요약 파일이 없습니다: $([System.Web.HttpUtility]::HtmlEncode($CsvPath))</p>" }
  $rows = Import-Csv $CsvPath | Select-Object -First $Top
  if (-not $rows){ return "<p>표시할 데이터가 없습니다.</p>" }
  $cols = $rows[0].PSObject.Properties.Name
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine('<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-family:Segoe UI,Arial,sans-serif;font-size:12px;">')
  [void]$sb.AppendLine('<thead><tr>')
  foreach($c in $cols){ [void]$sb.AppendLine("<th style='background:#f2f2f2;text-align:left;'>$([System.Web.HttpUtility]::HtmlEncode($c))</th>") }
  [void]$sb.AppendLine('</tr></thead><tbody>')
  foreach($r in $rows){
    [void]$sb.AppendLine('<tr>')
    foreach($c in $cols){
      $val = [string]$r.$c
      [void]$sb.AppendLine("<td>$([System.Web.HttpUtility]::HtmlEncode($val))</td>")
    }
    [void]$sb.AppendLine('</tr>')
  }
  [void]$sb.AppendLine('</tbody></table>')
  $sb.ToString()
}

# ---------- Build subject/body ----------
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
  </ul>
  <h3>요약 미리보기 (상위 10행)</h3>
  $TopTableHtml
  <p style="margin-top:16px;">로그: $(Resolve-Path $EmailLog)</p>
</body>
</html>
"@

# ---------- Mail sender (System.Net.Mail; UTF-8 + HTML Base64) ----------
function Send-ReportMail {
  param(
    [Parameter(Mandatory)][string]$Subject,
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

  # 1) 메시지 생성
  $msg = New-Object System.Net.Mail.MailMessage
  if (-not $From) { throw "MAIL_FROM is empty" }
  $msg.From = $From
  foreach ($to in $ToList) { [void]$msg.To.Add($to) }

  # 2) 제목/헤더 인코딩
  $msg.Subject = $Subject
  $msg.SubjectEncoding = $enc
  if ($msg.PSObject.Properties.Name -contains 'HeadersEncoding') { $msg.HeadersEncoding = $enc }

  # 3) HTML 본문 (AlternateView + Base64)
  $alt = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($BodyHtml, $enc, "text/html")
  $alt.TransferEncoding = [System.Net.Mime.TransferEncoding]::Base64
  $msg.AlternateViews.Clear()
  $msg.AlternateViews.Add($alt)
  # 호환용(일부 클라이언트): Body/BodyEncoding도 세팅
  $msg.IsBodyHtml = $true
  $msg.Body        = $BodyHtml
  $msg.BodyEncoding = $enc

  # 4) 첨부 (파일명 UTF-8)
  foreach ($p in $Attachments) {
    if ($p -and (Test-Path -LiteralPath $p)) {
      $att = New-Object System.Net.Mail.Attachment($p)
      if ($att.PSObject.Properties.Name -contains 'NameEncoding') { $att.NameEncoding = $enc }
      $msg.Attachments.Add($att) | Out-Null
    }
  }

  # 5) SMTP
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
    -Subject $Subject `
    -BodyHtml $BodyHtml `
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