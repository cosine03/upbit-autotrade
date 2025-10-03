<# ===============================
 run_daily_report_and_email.ps1  (PowerShell 5.x safe)
 Purpose: load .env (optional), build daily HTML report, send email.
 Notes:
 - ASCII-only comments (avoid mojibake)
 - No console encoding tweaks
 - Uses Send-MailMessage (PS5 built-in)
 - Data sources: trades_closed.csv, rejects.csv
================================ #>

param(
  [int]    $SinceHours       = 24,
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
  [string] $SubjectPrefix    = "[PaperTrader] Daily Report",
  [ValidateSet('AM','PM')] [string]$TagHalf = 'AM'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve script root (works for scheduler/console/direct file run)
$ScriptRoot = if ($PSCommandPath) {
  Split-Path -Parent $PSCommandPath
} elseif ($MyInvocation.MyCommand.Path) {
  Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
  "D:\upbit_autotrade_starter"
}
Set-Location -Path $ScriptRoot

# -------- helpers --------
function New-NowUtc { [DateTimeOffset]::UtcNow }

function Load-DotEnv {
  param([Parameter(Mandatory)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return }
  (Get-Content -LiteralPath $Path -Encoding utf8) |
    Where-Object { $_ -match '^\s*[^#].*=' } |
    ForEach-Object {
      if ($_ -match '^\s*([^=]+)=(.*)$') {
        $name  = $matches[1].Trim()
        $value = $matches[2].Trim().Trim("'`"")
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
      }
    }
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

function Fmt6($n){
  if ($null -eq $n) { return "-" }
  return ("{0:N6}" -f [double]$n)
}

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
      elseif ($c -in @('entry_price','exit_price','pnl','fee')) {
        $val = if ($null -ne $val) { Fmt6 $val } else { "" }
      }
      $null = $sb.AppendLine("<td>$val</td>")
    }
    $null = $sb.AppendLine("</tr>")
  }
  $null = $sb.AppendLine("</tbody></table>")
  return $sb.ToString()
}

# -------- .env -> parameters (optional) --------
if ($UseEnv) {
  $envFile = Join-Path $ScriptRoot ".env"
  Load-DotEnv -Path $envFile

  if (-not $SmtpServer -and $env:SMTP_SERVER) { $SmtpServer = $env:SMTP_SERVER }
  if (-not $SmtpServer -and $env:SMTP_HOST)   { $SmtpServer = $env:SMTP_HOST }
  if (-not $From       -and $env:SMTP_FROM)   { $From       = $env:SMTP_FROM }
  if (-not $To         -and $env:SMTP_TO)     { $To         = $env:SMTP_TO }
  if (-not $SmtpUser   -and $env:SMTP_USER)   { $SmtpUser   = $env:SMTP_USER }
  if (-not $SmtpPass   -and $env:SMTP_PASS)   { $SmtpPass   = ConvertTo-SecureString $env:SMTP_PASS -AsPlainText -Force }
  if ($env:SMTP_PORT) { try { $SmtpPort   = [int]$env:SMTP_PORT } catch {} }
  if ($env:SMTP_TLS)  { try { $UseStartTls = [bool]::Parse($env:SMTP_TLS) } catch {} }
}

# plain string password -> SecureString
if ($SmtpPass -is [string] -and $SmtpPass) {
  $SmtpPass = ConvertTo-SecureString $SmtpPass -AsPlainText -Force
}

# required checks
if (-not $From -or -not $To -or -not $SmtpServer -or -not $SmtpUser -or -not $SmtpPass) {
  throw "Missing SMTP fields: From/To/Server/User/Pass (set via .env or parameters)."
}

# -------- optional pre-pipeline --------
if ($RunPipeline) {
  try {
    $pipe = Join-Path $ScriptRoot "run_pipeline.ps1"
    if (Test-Path $pipe) {
      if ($TagHalf) { & $pipe -TagHalf $TagHalf } else { & $pipe }
    }
  } catch {
    Write-Warning "[WARN] run_pipeline failed: $($_.Exception.Message)"
  }
}

# -------- load data --------
$cut     = (New-NowUtc).AddHours(-$SinceHours)
$trades  = Read-CsvSafe $TradesClosedCsv
$rejects = Read-CsvSafe $RejectsCsv

# -------- normalize trades_closed (fix quick_expired / mis-ordered columns) --------
$closedRows = @()
foreach($row in $trades){
  $opened_at = ToDate $row.opened_at
  $closed_at = ToDate $row.closed_at

  $entry0  = Parse-Double $row.entry_price
  $exit0   = Parse-Double $row.exit_price
  $pnl0    = Parse-Double $row.pnl
  $fee0    = Parse-Double $row.fee
  $reason0 = $row.reason

  # Case A: entry holds string(reason), exit holds fee, pnl empty
  $isEntryNumber = "$($row.entry_price)" -match '^\s*[\+\-]?\d+(\.\d+)?\s*$'
  if (($null -eq $entry0) -and ($null -ne (Parse-Double $row.exit_price)) -and -not $isEntryNumber -and ($null -eq $pnl0)) {
    $reason0 = $row.entry_price
    $fee0    = Parse-Double $row.exit_price
    $entry0  = $null; $exit0 = $null; $pnl0 = $null
  }

  # Case B: reason is numeric (actually pnl), pnl holds exit, exit holds entry
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

# filter window and sort (force array)
$closedRows = @($closedRows | Where-Object { $_.closed_at -and $_.closed_at -ge $cut } | Sort-Object closed_at)

# -------- simple stats (array-safe) --------
$cnt      = @($closedRows).Count
$pnlVals  = @($closedRows | ForEach-Object { $_.pnl } | Where-Object { $_ -ne $null })
$totalPnl = if ($pnlVals.Count -gt 0){ ($pnlVals | Measure-Object -Sum).Sum } else { 0 }

$winRows  = @($closedRows | Where-Object { $_.pnl -gt 0 })
$lossRows = @($closedRows | Where-Object { $_.pnl -lt 0 })
$flatRows = @($closedRows | Where-Object { $_.pnl -eq 0 -or $_.pnl -eq $null })

$winCnt   = $winRows.Count
$lossCnt  = $lossRows.Count
$flatCnt  = $flatRows.Count

$avgWin   = if ($winRows.Count  -gt 0){ ($winRows  | Measure-Object -Property pnl -Average).Average } else { $null }
$avgLoss  = if ($lossRows.Count -gt 0){ ($lossRows | Measure-Object -Property pnl -Average).Average } else { $null }

# -------- rejects summary --------
$rejRows =
  @($rejects |
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
  Where-Object { $_.now -and $_.now -ge $cut })

$rejSummary =
  @($rejRows |
  Group-Object reason |
  Sort-Object Count -Desc |
  ForEach-Object { [pscustomobject]@{ reason = $_.Name; count = $_.Count } })

# -------- HTML --------
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
$subject  = "$SubjectPrefix$tag | last ${SinceHours}h (UTC)"

$summaryHtml = @"
$style
<h2>PaperTrader Daily Report</h2>
<p class="small">Window: <strong>$sinceStr</strong> -> <strong>$nowStr</strong> (UTC)</p>
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

# -------- send mail --------
$toList = $To -split '\s*,\s*' | Where-Object { $_ -and $_.Trim() -ne "" }
if (-not $SmtpUser) { $SmtpUser = $From }
if (-not $SmtpPass) { $SmtpPass = Read-Host -AsSecureString -Prompt "SMTP app password for $SmtpUser" }
$cred = New-Object System.Management.Automation.PSCredential($SmtpUser, $SmtpPass)

$attachments = @()
if ($AttachCsv -and (Test-Path $TradesClosedCsv)) { $attachments += (Resolve-Path $TradesClosedCsv).Path }

Send-MailMessage `
  -From $From -To $toList -Subject $subject `
  -Body $htmlBody -BodyAsHtml `
  -SmtpServer $SmtpServer -Port $SmtpPort -UseSsl:$UseStartTls `
  -Credential $cred `
  -Attachments $attachments `
  -ErrorAction Stop

Write-Host "[OK] Mail sent to $($toList -join ', ') ($subject)"