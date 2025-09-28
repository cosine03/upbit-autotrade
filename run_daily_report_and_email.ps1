<# 
run_daily_report_and_email.ps1  (DEDUP + UTF8 메일)

기능
- AM/PM별 리포트 생성(옵션) + 메일 발송
- trades 두 전략 결과를 merge → 중복제거(dedup) → 요약 재계산(이 dedup본 기준)
- 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log

첨부
 1) bt_stats_summary_merged_{AM|PM}.csv        (dedup 기반 요약)
 2) bt_breakout_only\bt_tv_events_stats_summary.csv
 3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv
 4) bt_trades_merged_dedup_{AM|PM}.csv         (참고용, 선택 첨부)

.env 예(Gmail)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password_here     # 앱 비밀번호
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
$Root       = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$DATE       = Get-Date -Format "yyyy-MM-dd"

$DailyDir        = Join-Path $Root "logs\daily\${DATE}_$TagHalf"
$ReportsDir      = Join-Path $Root "logs\reports"
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

# ---------- Logging ----------
$EmailLog = Join-Path $ReportsDir ("email_{0}_{1}.log" -f $DATE, $TagHalf)
function Write-Log([string]$msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[{0}] {1}" -f $stamp, $msg
  $line | Tee-Object -FilePath $EmailLog -Append
}

Write-Log "== RUN START == Root=$Root  Half=$TagHalf  ==="
Write-Log "DATE            : $DATE"
Write-Log "DailyDir        : $DailyDir"

# ---------- .env loader ----------
$DotEnv = Join-Path $Root ".env"
function Load-DotEnv($path) {
  if (Test-Path -LiteralPath $path) {
    Get-Content $path | ForEach-Object {
      $line = $_.Trim()
      if (-not $line) { return }
      if ($line -match '^\s*#') { return }
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

# ---------- Key files & dirs ----------
$SignalsTV        = Join-Path $Root "logs\signals_tv.csv"
$BtDirBreakout    = Join-Path $DailyDir "bt_breakout_only"
$BtDirBoxLine     = Join-Path $DailyDir "bt_boxin_linebreak"

$BreakoutTrades   = Join-Path $BtDirBreakout "bt_tv_events_trades.csv"
$BoxLineTrades    = Join-Path $BtDirBoxLine  "bt_tv_events_trades.csv"

$BreakoutSummary  = Join-Path $BtDirBreakout "bt_tv_events_stats_summary.csv"
$BoxLineSummary   = Join-Path $BtDirBoxLine  "bt_tv_events_stats_summary.csv"

$MergedTradesDedup = Join-Path $DailyDir ("bt_trades_merged_dedup_{0}.csv" -f $TagHalf)
$MergedSummary     = Join-Path $DailyDir ("bt_stats_summary_merged_{0}.csv" -f $TagHalf)

Write-Log "BreakoutTrades  : $BreakoutTrades"
Write-Log "BoxLineTrades   : $BoxLineTrades"
Write-Log "MergedTrades(d) : $MergedTradesDedup"
Write-Log "MergedSummary   : $MergedSummary"

# ---------- Optional: run upstream pipeline (신호 분리/백테스트 등) ----------
if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }
    # (필요 시: 백테스트/전처리 호출을 여기에 배치)
    Write-Log "[PIPE] done"
  } catch { Write-Log "[PIPE][ERROR] $($_.Exception.Message)" }
}

# ---------- Merge + DEDUP trades ----------
function Read-CsvSafe($path) {
  if (Test-Path -LiteralPath $path) { return Import-Csv $path } else { return @() }
}

$trA = Read-CsvSafe $BreakoutTrades
$trB = Read-CsvSafe $BoxLineTrades

# 전략 라벨 부여
$trA | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force }
$trB | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }

$trAll = @($trA + $trB)

if ($trAll.Count -eq 0) {
  Write-Log "[DEDUP][WARN] no trades found; skip dedup & summary."
} else {
  # 숫자형 캐스팅 보정
  foreach ($r in $trAll) {
    foreach ($numCol in @('expiry_h','net')) {
      if ($r.PSObject.Properties.Name -contains $numCol) {
        if ($r.$numCol -ne $null -and $r.$numCol.ToString() -ne '') {
          $r.$numCol = [double]$r.$numCol
        } else { $r.$numCol = [double]0 }
      }
    }
  }

  # ---------- DEDUP 키 정의 (존재하는 컬럼만 사용)
  $preferCols = @('symbol','event','expiry_h','ts_sig','side','entry_mode')
  $keyCols = @()
  foreach ($c in $preferCols) { if ($trAll[0].PSObject.Properties.Name -contains $c) { $keyCols += $c } }
  if ($keyCols.Count -eq 0) {
    # 최소 안정키
    $keyCols = @('event','expiry_h')
  }
  Write-Log ("[DEDUP] key = " + ($keyCols -join ', '))

  # 해시셋으로 첫 등장만 살림
  $seen = New-Object 'System.Collections.Generic.HashSet[string]'
  $dedup = New-Object System.Collections.Generic.List[object]
  foreach ($row in $trAll) {
    $key = ($keyCols | ForEach-Object { ($row.$_).ToString() }) -join '||'
    if ($seen.Add($key)) { [void]$dedup.Add($row) }
  }

  # 저장
  $dedup | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedTradesDedup
  Write-Log ("[DEDUP] saved -> {0} (rows={1})" -f $MergedTradesDedup, $dedup.Count)

  # ---------- DEDUP 기반 요약 재계산 ----------
  # group by (event, expiry_h)
  $groupMap = @{}
  foreach ($r in $dedup) {
    $gk = "{0}||{1}" -f $r.event, $r.expiry_h
    if (-not $groupMap.ContainsKey($gk)) { $groupMap[$gk] = New-Object System.Collections.Generic.List[object] }
    $groupMap[$gk].Add($r)
  }

  $rows = New-Object System.Collections.Generic.List[object]
  foreach ($kv in $groupMap.GetEnumerator()) {
    $parts = $kv.Key.Split('||',2)
    $ev = $parts[0]
    $ex = [double]$parts[1]

    $list = $kv.Value
    $nets = @()
    $wins = 0
    foreach ($rr in $list) {
      $nets += [double]$rr.net
      if ([double]$rr.net -gt 0) { $wins++ }
    }
    $trades = $list.Count
    if ($trades -eq 0) { continue }
    $avg = ($nets | Measure-Object -Average).Average
    $sum = ($nets | Measure-Object -Sum).Sum

    # median
    $sorted = $nets | Sort-Object
    $n = $sorted.Count
    if ($n -eq 0) { $median = 0 }
    elseif ($n % 2 -eq 1) { $median = [double]$sorted[([int][math]::Floor($n/2))] }
    else { $median = ([double]$sorted[$n/2 - 1] + [double]$sorted[$n/2]) / 2.0 }

    $win_rate = [double]$wins / [double]$trades

    $obj = [PSCustomObject]@{
      event       = $ev
      expiry_h    = [double]$ex
      trades      = $trades
      win_rate    = $win_rate
      avg_net     = $avg
      median_net  = $median
      total_net   = $sum
    }
    $rows.Add($obj) | Out-Null
  }

  # expiry_h, event 순 정렬 후 저장
  $rows | Sort-Object @{Expression='event'},{Expression='expiry_h'} |
    Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
  Write-Log "merged summary saved -> $MergedSummary"
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

# ---------- Mail: subject/body/attachments ----------
$subject = "[Autotrade] Daily Report $DATE $TagHalf"

# 간단 텍스트 + 파일명 포함
$body = @"
자동 생성 리포트 ($DATE $TagHalf)
첨부:
  - $(Split-Path $MergedSummary -Leaf)
  - $(Split-Path $BreakoutSummary -Leaf)
  - $(Split-Path $BoxLineSummary -Leaf)
  - $(Split-Path $MergedTradesDedup -Leaf)
로그: $(Resolve-Path $EmailLog)
"@

# 수신자 배열
$ToList = @()
if ($MAIL_TO) {
  $MAIL_TO.Split(',;') | ForEach-Object { $addr = $_.Trim(); if ($addr) { $ToList += $addr } }
}
if (-not $ToList -or $ToList.Count -eq 0) { Write-Log "[MAIL][ERROR] MAIL_TO empty."; throw "MAIL_TO empty" }

# 첨부 확인
$Attachments = @()
foreach ($p in @($MergedSummary,$BreakoutSummary,$BoxLineSummary,$MergedTradesDedup)) {
  if (Test-Path $p) { $Attachments += (Resolve-Path $p).Path }
  else { Write-Log "[ATTACH][WARN] not found -> $p" }
}

# ---------- Mail sender (System.Net.Mail; UTF-8/Quoted-Printable) ----------
function Send-ReportMail {
  param(
    [Parameter(Mandatory)][string]$Subject,
    [Parameter(Mandatory)][string]$Body,
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

  $msg.Subject         = $Subject
  $msg.SubjectEncoding = $enc
  $msg.IsBodyHtml      = $false

  # 텍스트 본문(UTF-8, Quoted-Printable)
  $alt = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($Body, $enc, "text/plain")
  $alt.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
  $msg.AlternateViews.Clear()
  $msg.AlternateViews.Add($alt)

  # 호환용 Body에도 세팅
  $msg.Body         = $Body
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
    -Subject $subject `
    -Body $body `
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