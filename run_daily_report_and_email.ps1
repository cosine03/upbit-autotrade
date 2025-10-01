<# ===============================
 run_daily_report_and_email.ps1
 - PaperTrader 데일리 리포트 메일 전송
 - 새 trades_closed.csv 헤더에 맞춤
 - PowerShell 5+ 가정, Send-MailMessage 사용
================================ #>

param(
  [string]$Root = (Resolve-Path ".").Path,
  [int]$SinceHours = 24,                    # 최근 N시간만 집계 (기본 24h)
  [string]$CsvPath = ".\logs\paper\trades_closed.csv",
  [string]$SignalsCsv = ".\logs\signals_tv.csv",
  [string]$RejectsCsv = ".\logs\paper\rejects.csv",

  # 메일 설정
  [string]$SmtpServer = "smtp.gmail.com",
  [int]$SmtpPort = 587,
  [string]$From = "you@example.com",
  [string]$To = "you@example.com",
  [string]$SubjectPrefix = "[PaperTrader] Daily Report",
  [securestring]$SmtpPassword,              # 없는 경우 프롬프트
  [switch]$AttachCsv                         # trades_closed.csv 첨부 여부
)

function New-NowUtc {
  return [datetimeoffset]::UtcNow
}

function Read-CsvSafe {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return @() }
  try {
    return Import-Csv $Path -ErrorAction Stop
  } catch {
    Write-Warning "[WARN] CSV 읽기 실패: $Path - $($_.Exception.Message)"
    return @()
  }
}

# 숫자/시간 변환 헬퍼
function ToDate($s) { try { [datetimeoffset]$s } catch { $null } }
function ToDouble($s) { if ($null -eq $s -or $s -eq "") { $null } else { [double]$s } }

# HTML 테이블 생성 (간단 버전)
function New-HtmlTable {
  param(
    [Parameter(Mandatory=$true)] [array]$Rows,
    [string[]]$Columns,
    [string]$Title = ""
  )
  if (-not $Rows -or $Rows.Count -eq 0) {
    return "<h3>$Title</h3><p><em>(데이터 없음)</em></p>"
  }
  if (-not $Columns) { $Columns = $Rows[0].psobject.Properties.Name }

  $ths = ($Columns | ForEach-Object { "<th>$($_)</th>" }) -join ""
  $trs = foreach($r in $Rows) {
    $tds = foreach($c in $Columns) {
      $val = $r.$c
      if ($val -is [datetimeoffset]) { $val = $val.ToString("u") }
      "<td>$val</td>"
    }
    "<tr>$($tds -join '')</tr>"
  }
  return @"
<h3>$Title</h3>
<table border="1" cellspacing="0" cellpadding="4">
  <thead><tr>$ths</tr></thead>
  <tbody>
    $($trs -join "`n")
  </tbody>
</table>
"@
}

# 1) 데이터 로드
$trades = Read-CsvSafe -Path $CsvPath
$signals = Read-CsvSafe -Path $SignalsCsv
$rejects = Read-CsvSafe -Path $RejectsCsv

# 2) 최근 N시간 필터 기준 시각
$cut = (New-NowUtc()).AddHours(-$SinceHours)

# 3) trades_closed 정규화 (새 헤더 가정)
#    opened_at, symbol, event, side, level, closed_at, entry_price, exit_price, pnl, reason, fee
$closedRows =
  $trades |
  ForEach-Object {
    [pscustomobject]@{
      opened_at   = (ToDate $_.opened_at)
      symbol      = $_.symbol
      event       = $_.event
      side        = $_.side
      level       = $_.level
      closed_at   = (ToDate $_.closed_at)
      entry_price = (ToDouble $_.entry_price)
      exit_price  = (ToDouble $_.exit_price)
      pnl         = (ToDouble $_.pnl)
      reason      = $_.reason
      fee         = (ToDouble $_.fee)
    }
  } |
  Where-Object { $_.closed_at -and ($_.closed_at -ge $cut) }

# 4) 요약 통계
$cnt = $closedRows.Count
$pnlVals = $closedRows | ForEach-Object { $_.pnl } | Where-Object { $_ -ne $null }
$totalPnl = if ($pnlVals) { ($pnlVals | Measure-Object -Sum).Sum } else { 0 }

$winRows = $closedRows | Where-Object { $_.pnl -gt 0 }
$lossRows = $closedRows | Where-Object { $_.pnl -lt 0 }
$flatRows = $closedRows | Where-Object { $_.pnl -eq 0 }

$winCnt = $winRows.Count
$lossCnt = $lossRows.Count
$flatCnt = $flatRows.Count

$avgWin = if ($winRows) { ($winRows | Measure-Object -Property pnl -Average).Average } else { $null }
$avgLoss = if ($lossRows) { ($lossRows | Measure-Object -Property pnl -Average).Average } else { $null }

# 5) 리젝트/시그널(선택) 최근 요약
$rejCut = $cut
$rejRows =
  $rejects |
  ForEach-Object {
    $now = ToDate $_.now
    $ts  = ToDate $_.ts
    [pscustomobject]@{
      now    = $now
      ts     = $ts
      symbol = $_.symbol
      event  = $_.event
      reason = $_.reason
      phase  = $_.phase
    }
  } |
  Where-Object { $_.now -and $_.now -ge $rejCut }

$rejSummary =
  $rejRows |
  Group-Object reason |
  Sort-Object Count -Descending |
  ForEach-Object {
    [pscustomobject]@{ reason = $_.Name; count = $_.Count }
  }

# 6) HTML 본문 생성
$sinceStr = $cut.ToString("u")
$nowStr = (New-NowUtc()).ToString("u")
$subject = "$SubjectPrefix — last ${SinceHours}h (UTC)"

$style = @"
<style>
body{font-family:Segoe UI,Arial,Helvetica,sans-serif;font-size:14px}
h2{margin:0 0 8px 0}
h3{margin:12px 0 6px 0}
table{border-collapse:collapse}
th,td{border:1px solid #ccc;padding:4px 6px}
.code{font-family:Consolas,monospace;background:#f6f6f6;border:1px solid #e0e0e0;padding:6px}
.small{color:#666}
</style>
"@

$summaryHtml = @"
$style
<h2>PaperTrader Daily Report</h2>
<p class="small">Window: <strong>$sinceStr</strong> → <strong>$nowStr</strong> (UTC)</p>

<h3>PNL Summary</h3>
<ul>
  <li>Total closed trades: <strong>$cnt</strong></li>
  <li>Total PnL: <strong>$("{0:N6}" -f $totalPnl)</strong></li>
  <li>Wins / Losses / Flats: <strong>$winCnt</strong> / <strong>$lossCnt</strong> / <strong>$flatCnt</strong></li>
  <li>Avg Win: <strong>$("{0:N6}" -f ($avgWin ?? 0))</strong>, Avg Loss: <strong>$("{0:N6}" -f ($avgLoss ?? 0))</strong></li>
</ul>
"@

# 상세 표들
$closedTable = New-HtmlTable -Rows $closedRows -Title "Closed Trades (last ${SinceHours}h)" -Columns @(
  'opened_at','closed_at','symbol','event','side','level','entry_price','exit_price','pnl','reason','fee'
)

$rejTable = New-HtmlTable -Rows $rejSummary -Title "Rejects by Reason (last ${SinceHours}h)" -Columns @('reason','count')

$htmlBody = $summaryHtml + $closedTable + $rejTable

# 7) 메일 전송
if (-not $SmtpPassword) {
  $SmtpPassword = Read-Host -AsSecureString -Prompt "SMTP password for $From"
}
$cred = New-Object System.Management.Automation.PSCredential($From, $SmtpPassword)

# Send-MailMessage는 기본적으로 ANSI 본문으로 나가므로, -BodyAsHtml 로 HTML 처리
# 일부 환경에서 인코딩 이슈가 있으면 SmtpClient 직접 써서 UTF8 지정하는 버전으로 교체 가능.
$attachments = @()
if ($AttachCsv -and (Test-Path $CsvPath)) { $attachments += (Resolve-Path $CsvPath).Path }

Send-MailMessage `
  -From $From `
  -To $To `
  -Subject $subject `
  -Body $htmlBody `
  -BodyAsHtml `
  -SmtpServer $SmtpServer `
  -Port $SmtpPort `
  -UseSsl `
  -Credential $cred `
  -Attachments $attachments `
  -ErrorAction Stop

Write-Host "[OK] Mail sent to $To ($subject)"