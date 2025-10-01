<# ===============================
 run_daily_report_and_email.ps1 (PS5-stable)
 - .env 로드(-UseEnv)
 - run_pipeline.ps1 사전 실행(-RunPipeline)
 - trades_closed 레거시 행 교정(A/B)
 - HTML 메일 + CSV 첨부
 표준 헤더:
 opened_at,symbol,event,side,level,closed_at,entry_price,exit_price,pnl,reason,fee
================================ #>

<# ===============================
 run_daily_report_and_email.ps1 (PS5-stable)
 - .env 로드(-UseEnv)
 - run_pipeline.ps1 사전 실행(-RunPipeline)
 - trades_closed 레거시 행 교정(A/B)
 - HTML 메일 + CSV 첨부
 표준 헤더:
 opened_at,symbol,event,side,level,closed_at,entry_price,exit_price,pnl,reason,fee
================================ #>

# 반드시 스크립트의 첫 번째 구문이어야 함
param(
  [int]    $SinceHours       = 24,
  [string] $TagHalf          = 'AM',
  [switch] $RunPipeline,
  [switch] $AttachCsv,
  [switch] $UseEnv,

  [string] $SignalsCsv       = ".\logs\signals_tv.csv",
  [string] $RejectsCsv       = ".\logs\paper\rejects.csv",
  [string] $TradesOpenCsv    = ".\logs\paper\trades_open.csv",
  [string] $TradesClosedCsv  = ".\logs\paper\trades_closed.csv",

  [string] $SmtpServer,
  [int]    $SmtpPort         = 587,
  [string] $From,
  [string] $To,
  [string] $SmtpUser,
  [Security.SecureString] $SmtpPass,
  [bool]   $UseStartTls      = $true,

  [string] $SubjectPrefix    = "[PaperTrader] Daily Report"
)

# ← 여기부터는 다른 구문 가능
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($true)
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

# ---------- 유틸 ----------
function New-NowUtc { [DateTimeOffset]::UtcNow }

function Load-DotEnv([string]$Path){
  if (-not (Test-Path $Path)) { Write-Warning "[WARN] .env not found: $Path"; return }
  $raw = Get-Content -Path $Path -Raw -Encoding UTF8
  if ($raw.Length -gt 0 -and $raw[0] -eq [char]0xFEFF) { $raw = $raw.Substring(1) } # BOM 제거
  foreach($line in $raw -split "`n"){
    $line = $line.Trim()
    if (-not $line -or $line.StartsWith('#')) { continue }
    $eq = $line.IndexOf('=')
    if ($eq -lt 1) { continue }
    $k = $line.Substring(0,$eq).Trim()
    $v = $line.Substring($eq+1).Trim()
    if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1,$v.Length-2) }
    if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1,$v.Length-2) }
    Set-Item -Path ("Env:{0}" -f $k) -Value $v
  }
  Write-Host "[INFO] .env loaded (SMTP_* / MAIL_*):" -ForegroundColor Cyan
  Get-ChildItem Env:SMTP_* , Env:MAIL_* | Sort-Object Name | Format-Table Name,Value -Auto
}

function Parse-Double($s){
  if ($null -eq $s) { return $null }
  $txt = "$s".Trim()
  if ($txt -eq "") { return $null }
  $out = 0.0
  if ([double]::TryParse($txt, [ref]$out)) { return $out } else { return $null }
}

function ToDate($s){
  if ($null -eq $s) { return $null }
  $txt = "$s".Trim()
  if ($txt -eq "") { return $null }
  try { return [datetimeoffset]::Parse($txt) } catch { return $null }
}

function Fmt6($n){ if ($null -eq $n) { "-" } else { "{0:N6}" -f [double]$n } }

function Read-CsvSafe([string]$Path){
  if (-not (Test-Path $Path)) { return @() }
  try { return Import-Csv $Path } catch { return @() }
}

function New-HtmlTable([array]$Rows,[string]$Title,[string[]]$Columns){
  $sb = New-Object System.Text.StringBuilder
  $null = $sb.AppendLine("<h3>$Title</h3>")
  if (-not $Rows -or $Rows.Count -eq 0) {
    $null = $sb.AppendLine("<p class=""small"">no rows</p>")
    return $sb.ToString()
  }
  $null = $sb.AppendLine("<table>")
  $null = $sb.AppendLine("<thead><tr>")
  foreach($c in $Columns){ $null = $sb.AppendLine("<th>$c</th>") }
  $null = $sb.AppendLine("</tr></thead><tbody>")
  foreach($r in $Rows){
    $null = $sb.AppendLine("<tr>")
    foreach($c in $Columns){
      $val = $r.$c
      if ($val -is [datetimeoffset]) { $val = $val.ToString("u") }
      elseif ($c -in @('entry_price','exit_price','pnl','fee')) { $val = if ($null -ne $val) { Fmt6 $val } else { "" } }
      $null = $sb.AppendLine("<td>$val</td>")
    }
    $null = $sb.AppendLine("</tr>")
  }
  $null = $sb.AppendLine("</tbody></table>")
  return $sb.ToString()
}

# ---------- .env -> 파라미터 보강 ----------
if ($UseEnv) {
  $root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
  Load-DotEnv -Path (Join-Path $root ".env")

  # 우선순위: SMTP_* -> MAIL_*(별칭)
  if (-not $SmtpServer) { if ($env:SMTP_SERVER) { $SmtpServer = $env:SMTP_SERVER } elseif ($env:SMTP_HOST) { $SmtpServer = $env:SMTP_HOST } }
  if (-not $From)       { if ($env:SMTP_FROM)   { $From   = $env:SMTP_FROM }   elseif ($env:MAIL_FROM) { $From = $env:MAIL_FROM } }
  if (-not $To)         { if ($env:SMTP_TO)     { $To     = $env:SMTP_TO }     elseif ($env:MAIL_TO)   { $To   = $env:MAIL_TO } }
  if (-not $SmtpUser)   { if ($env:SMTP_USER)   { $SmtpUser = $env:SMTP_USER } }
  if (-not $SmtpPass)   { if ($env:SMTP_PASS)   { $SmtpPass = ConvertTo-SecureString $env:SMTP_PASS -AsPlainText -Force } }
  if ($env:SMTP_PORT) { try { $SmtpPort = [int]$env:SMTP_PORT } catch {} }
  if ($env:SMTP_TLS)  { try { $UseStartTls = [bool]::Parse($env:SMTP_TLS) } catch {} }
}

# 평문 pw로 넘어오면 SecureString 변환
if ($SmtpPass -is [string] -and $SmtpPass) { $SmtpPass = ConvertTo-SecureString $SmtpPass -AsPlainText -Force }

# 필수 체크
if (-not $From -or -not $To -or -not $SmtpServer -or -not $SmtpUser -or -not $SmtpPass) {
  throw "Missing SMTP From/To/Server/User/Pass (use .env or pass as parameters)."
}

# ---------- (옵션) 사전 파이프라인 ----------
if ($RunPipeline) {
  try {
    $root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    $pipe = Join-Path $root "run_pipeline.ps1"
    if (Test-Path $pipe) {
      if ($TagHalf) { & $pipe -TagHalf $TagHalf } else { & $pipe }
      Write-Host "[OK] Pre-pipeline executed." -ForegroundColor Green
    } else {
      Write-Warning "[WARN] run_pipeline.ps1 없음 - 건너뜀"
    }
  } catch { Write-Warning "[WARN] Pre-pipeline failed: $($_.Exception.Message)" }
}

# ---------- 데이터 로드 ----------
$cut     = (New-NowUtc).AddHours(-$SinceHours)
$trades  = Read-CsvSafe $TradesClosedCsv
$rejects = Read-CsvSafe $RejectsCsv

# ---------- trades_closed 정규화(+레거시 교정) ----------
$closedRows = @()
foreach($row in $trades){
  $opened_at = ToDate $row.opened_at
  $closed_at = ToDate $row.closed_at

  $entry0  = Parse-Double $row.entry_price
  $exit0   = Parse-Double $row.exit_price
  $pnl0    = Parse-Double $row.pnl
  $fee0    = Parse-Double $row.fee
  $reason0 = $row.reason

  # 패턴 A: entry=quick_expired(문자), exit=0.001(수치), pnl 비어있음
  $isEntryNumber = $row.entry_price -match '^\s*[\+\-]?\d+(\.\d+)?\s*$'
  if (($null -eq $entry0) -and ($null -ne (Parse-Double $row.exit_price)) -and -not $isEntryNumber -and ($null -eq $pnl0)) {
    $reason0 = $row.entry_price
    $fee0    = Parse-Double $row.exit_price
    $entry0  = $null; $exit0 = $null; $pnl0 = $null
  }

  # 패턴 B: reason 자리에 pnl(숫자) 밀려온 케이스
  $reasonAsNum = Parse-Double $row.reason
  if (($null -eq $entry0) -and ($null -ne $exit0) -and ($null -ne $pnl0) -and ($null -ne $reasonAsNum) -and ($null -ne $fee0)) {
    $entry0  = $exit0
    $exit0   = $pnl0
    $pnl0    = $reasonAsNum
    $reason0 = "time_expired"
  }

  $closedRows += [pscustomobject]@{
    opened_at   = $opened_at
    symbol      = $row.symbol
    event       = $row.event
    side        = $row.side
    level       = $row.level
    closed_at   = $closed_at
    entry_price = $entry0
    exit_price  = $exit0
    pnl         = $pnl0
    reason      = $reason0
    fee         = $fee0
  }
}
$closedRows = $closedRows | Where-Object { $_.closed_at -and $_.closed_at -ge $cut } | Sort-Object closed_at

# ---------- 요약 ----------
$cnt      = $closedRows.Count
$pnlVals  = $closedRows | ForEach-Object { $_.pnl } | Where-Object { $_ -ne $null }
$totalPnl = if ($pnlVals){ ($pnlVals | Measure-Object -Sum).Sum } else { 0 }
$winRows  = $closedRows | Where-Object { $_.pnl -gt 0 }
$lossRows = $closedRows | Where-Object { $_.pnl -lt 0 }
$flatRows = $closedRows | Where-Object { $_.pnl -eq 0 -or $_.pnl -eq $null }
$winCnt   = $winRows.Count; $lossCnt = $lossRows.Count; $flatCnt = $flatRows.Count
$avgWin   = if ($winRows){ ($winRows | Measure-Object -Property pnl -Average).Average } else { $null }
$avgLoss  = if ($lossRows){ ($lossRows | Measure-Object -Property pnl -Average).Average } else { $null }

# ---------- 리젝트 요약 ----------
$rejRows =
  $rejects |
  ForEach-Object {
    [pscustomobject]@{
      now    = (ToDate $_.now)
      ts     = (ToDate $_.ts)
      symbol = $_.symbol
      event  = $_.event
      reason = $_.reason
      phase  = $_.phase
    }
  } |
  Where-Object { $_.now -and $_.now -ge $cut }

$rejSummary =
  $rejRows |
  Group-Object reason |
  Sort-Object Count -Desc |
  ForEach-Object { [pscustomobject]@{ reason = $_.Name; count = $_.Count } }

# ---------- HTML ----------
$style = @"
<style>
body{font-family:Segoe UI,Arial,Helvetica,sans-serif;font-size:14px}
h2{margin:0 0 8px 0}
h3{margin:12px 0 6px 0}
table{border-collapse:collapse}
th,td{border:1px solid #ccc;padding:4px 6px}
.small{color:#666}
</style>
"@

$sinceStr = $cut.ToString("u")
$nowStr   = (New-NowUtc).ToString("u")
$tag      = if ($TagHalf) { " $TagHalf" } else { "" }
$subject  = "$SubjectPrefix$tag — last ${SinceHours}h (UTC)"

$summaryHtml = @"
$style
<h2>PaperTrader Daily Report</h2>
<p class="small">Window: <strong>$sinceStr</strong> → <strong>$nowStr</strong> (UTC)</p>
<h3>PNL Summary</h3>
<ul>
  <li>Total closed trades: <strong>$cnt</strong></li>
  <li>Total PnL: <strong>$(Fmt6 $totalPnl)</strong></li>
  <li>Wins / Losses / Flats: <strong>$winCnt</strong> / <strong>$lossCnt</strong> / <strong>$flatCnt</strong></li>
  <li>Avg Win: <strong>$(Fmt6 $avgWin)</strong>, Avg Loss: <strong>$(Fmt6 $avgLoss)</strong></li>
</ul>
"@

$closedTable = New-HtmlTable -Rows $closedRows -Title "Closed Trades (last ${SinceHours}h)" -Columns @(
  'opened_at','closed_at','symbol','event','side','level','entry_price','exit_price','pnl','reason','fee'
)
$rejTable    = New-HtmlTable -Rows $rejSummary -Title "Rejects by Reason (last ${SinceHours}h)" -Columns @('reason','count')
$htmlBody    = $summaryHtml + $closedTable + $rejTable

# ---------- 메일 ----------
if (-not $SmtpUser) { $SmtpUser = $From }
if (-not $SmtpPass) { $SmtpPass = Read-Host -AsSecureString -Prompt "SMTP app password for $SmtpUser" }
$cred = New-Object System.Management.Automation.PSCredential($SmtpUser, $SmtpPass)

# To가 "a@x,b@y" 형태면 반드시 배열로 쪼갬(헤더 콤마 오류 방지)
$toList = $To -split '\s*,\s*'

$attachments = @()
if ($AttachCsv -and (Test-Path $TradesClosedCsv)) { $attachments += (Resolve-Path $TradesClosedCsv).Path }

Send-MailMessage `
  -From $From -To $toList -Subject $subject `
  -Body $htmlBody -BodyAsHtml `
  -SmtpServer $SmtpServer -Port $SmtpPort -UseSsl:$UseStartTls `
  -Credential $cred `
  -Attachments $attachments `
  -ErrorAction Stop

Write-Host "[OK] Mail sent to $To ($subject)" -ForegroundColor Green