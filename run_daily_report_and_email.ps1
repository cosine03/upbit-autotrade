<# 
run_daily_report_and_email.ps1  (dedup + overlap logging)

- AM/PM별 일일 리포트 생성(옵션) + 메일 발송
- 병합 요약: bt_stats_summary_merged_{AM|PM}.csv
- dedup 트레이드: bt_trades_merged_dedup_{AM|PM}.csv
- 메일 전송 로그: logs\reports\email_YYYY-MM-DD_{AM|PM}.log
- 첨부: 
  1) bt_stats_summary_merged_{AM|PM}.csv
  2) bt_breakout_only\bt_tv_events_stats_summary.csv
  3) bt_boxin_linebreak\bt_tv_events_stats_summary.csv

필수: 루트 경로(.env, logs, scripts)가 이 스크립트와 같은 폴더에 있다고 가정.

.env 예(Gmail):
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password_here   # 앱 비밀번호(일반 비번 X)
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
Write-Log "== RUN START == Root=$Root  Half=$TagHalf =="

# ---------- .env loader ----------
$DotEnv = Join-Path $Root ".env"
function Load-DotEnv($path) {
  if (Test-Path -LiteralPath $path) {
    Get-Content $path | ForEach-Object {
      $line = $_.Trim()
      if (-not $line) { return }                 # 빈 줄 skip
      if ($line -match '^\s*#') { return }       # 주석 줄 skip
      if ($line.Contains('#')) {                 # 라인 끝 주석 제거
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
    Write-Log "[WARN] .env not found at $path"
  }
}
Load-DotEnv $DotEnv

# ---------- Key files ----------
$BtDirBreakout   = Join-Path $DailyDir "bt_breakout_only"
$BtDirBoxLine    = Join-Path $DailyDir "bt_boxin_linebreak"

$BreakoutSummary = Join-Path $BtDirBreakout "bt_tv_events_stats_summary.csv"
$BoxLineSummary  = Join-Path $BtDirBoxLine  "bt_tv_events_stats_summary.csv"
$MergedSummary   = Join-Path $DailyDir ("bt_stats_summary_merged_{0}.csv" -f $TagHalf)

$BreakoutTrades  = Join-Path $BtDirBreakout "bt_tv_events_trades.csv"
$BoxLineTrades   = Join-Path $BtDirBoxLine  "bt_tv_events_trades.csv"
$DedupTradesOut  = Join-Path $DailyDir ("bt_trades_merged_dedup_{0}.csv" -f $TagHalf)

Write-Log "DailyDir       : $DailyDir"
Write-Log "BreakoutSummary: $BreakoutSummary"
Write-Log "BoxLineSummary : $BoxLineSummary"
Write-Log "MergedSummary  : $MergedSummary"
Write-Log "BreakoutTrades : $BreakoutTrades"
Write-Log "BoxLineTrades  : $BoxLineTrades"
Write-Log "DedupTradesOut : $DedupTradesOut"

# ---------- Optional: run pipeline before email ----------
if ($RunPipeline) {
  try {
    Write-Log "[PIPE] start"
    if (-not (Test-Path $DailyDir)) { New-Item -ItemType Directory -Force -Path $DailyDir | Out-Null }

    # (여기에 신호 분리/백테스트 호출을 넣을 수 있음. 지금은 merge만 수행)

    # (1) 요약 병합
    if ((Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
      $b = Import-Csv $BreakoutSummary
      $l = Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only"   -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log "merged summary saved -> $MergedSummary"
    } else {
      Write-Log "[PIPE][WARN] summary files missing; skip merge."
    }

    # (2) 트레이드 dedup + overlap 로깅
    if ((Test-Path $BreakoutTrades) -and (Test-Path $BoxLineTrades)) {
      $tA = Import-Csv $BreakoutTrades | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force; $_ }
      $tB = Import-Csv $BoxLineTrades  | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force; $_ }

      # 키 후보 (존재하는 컬럼만 사용)
      $prefCols = @('symbol','event','side','expiry_h','ts_sig','entry_ts')

      function Get-Key($row, $pref) {
        $have = @()
        foreach ($c in $pref) {
          if ($row.PSObject.Properties.Name -contains $c) {
            $have += ("{0}={1}" -f $c, ($row.$c))
          }
        }
        if ($have.Count -gt 0) { return ($have -join '|') }
        # 후보가 하나도 없으면 전체 json으로
        return (ConvertTo-Json $row -Compress)
      }

      # 키셋/중복 계산
      $setA = New-Object System.Collections.Generic.HashSet[string]
      $setB = New-Object System.Collections.Generic.HashSet[string]
      foreach ($r in $tA) { [void]$setA.Add((Get-Key $r $prefCols)) }
      foreach ($r in $tB) { [void]$setB.Add((Get-Key $r $prefCols)) }

      $inter = [Linq.Enumerable]::ToArray([Linq.Enumerable]::Intersect($setA, $setB))
      $sameA = ($setA.Count -eq $inter.Count) -and ($setB.Count -eq $inter.Count)

      Write-Log ("[DEDUP] breakout_only unique={0}, boxin_linebreak unique={1}, intersection={2}, identical_sets={3}" -f $setA.Count, $setB.Count, $inter.Count, $sameA)

      # 실제 dedup (A+B의 key 기반 유니크)
      $seen = New-Object System.Collections.Generic.HashSet[string]
      $out  = New-Object System.Collections.Generic.List[object]
      foreach ($r in ($tA + $tB)) {
        $k = Get-Key $r $prefCols
        if (-not $seen.Contains($k)) {
          [void]$seen.Add($k)
          [void]$out.Add($r)
        }
      }
      $out | Export-Csv -NoTypeInformation -Encoding UTF8 $DedupTradesOut
      Write-Log ("[DEDUP] saved -> {0} (rows={1})" -f $DedupTradesOut, $out.Count)
    } else {
      Write-Log "[DEDUP][WARN] trades files missing; skip dedup."
    }

    Write-Log "[PIPE] done"
  } catch {
    Write-Log "[PIPE][ERROR] $($_.Exception.Message)"
  }
} else {
  # 파이프라인 off: 병합 요약만 보장 시도
  if ((-not (Test-Path $MergedSummary)) -and (Test-Path $BreakoutSummary) -and (Test-Path $BoxLineSummary)) {
    try {
      $b = Import-Csv $BreakoutSummary
      $l = Import-Csv $BoxLineSummary
      $b | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only"   -Force }
      $l | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force }
      ($b + $l) | Export-Csv -NoTypeInformation -Encoding UTF8 $MergedSummary
      Write-Log "merged summary saved -> $MergedSummary"
    } catch {
      Write-Log "[MERGE][ERROR] $($_.Exception.Message)"
    }
  }

  # dedup도 가능하면 수행
  if ((Test-Path $BreakoutTrades) -and (Test-Path $BoxLineTrades)) {
    try {
      $tA = Import-Csv $BreakoutTrades | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "breakout_only" -Force; $_ }
      $tB = Import-Csv $BoxLineTrades  | ForEach-Object { $_ | Add-Member -NotePropertyName strategy -NotePropertyValue "boxin_linebreak" -Force; $_ }

      $prefCols = @('symbol','event','side','expiry_h','ts_sig','entry_ts')
      function Get-Key($row, $pref) {
        $have = @()
        foreach ($c in $pref) {
          if ($row.PSObject.Properties.Name -contains $c) {
            $have += ("{0}={1}" -f $c, ($row.$c))
          }
        }
        if ($have.Count -gt 0) { return ($have -join '|') }
        return (ConvertTo-Json $row -Compress)
      }

      $setA = New-Object System.Collections.Generic.HashSet[string]
      $setB = New-Object System.Collections.Generic.HashSet[string]
      foreach ($r in $tA) { [void]$setA.Add((Get-Key $r $prefCols)) }
      foreach ($r in $tB) { [void]$setB.Add((Get-Key $r $prefCols)) }
      $inter = [Linq.Enumerable]::ToArray([Linq.Enumerable]::Intersect($setA, $setB))
      $sameA = ($setA.Count -eq $inter.Count) -and ($setB.Count -eq $inter.Count)
      Write-Log ("[DEDUP] breakout_only unique={0}, boxin_linebreak unique={1}, intersection={2}, identical_sets={3}" -f $setA.Count, $setB.Count, $inter.Count, $sameA)

      $seen = New-Object System.Collections.Generic.HashSet[string]
      $out  = New-Object System.Collections.Generic.List[object]
      foreach ($r in ($tA + $tB)) {
        $k = Get-Key $r $prefCols
        if (-not $seen.Contains($k)) {
          [void]$seen.Add($k)
          [void]$out.Add($r)
        }
      }
      $out | Export-Csv -NoTypeInformation -Encoding UTF8 $DedupTradesOut
      Write-Log ("[DEDUP] saved -> {0} (rows={1})" -f $DedupTradesOut, $out.Count)
    } catch {
      Write-Log "[DEDUP][ERROR] $($_.Exception.Message)"
    }
  }
}

# ---------- Email config from .env ----------
$SMTP_HOST = if ($env:SMTP_HOST) { $env:SMTP_HOST } else { 'smtp.gmail.com' }
$SMTP_PORT = if ($env:SMTP_PORT) { [int]$env:SMTP_PORT } else { 587 }
$SMTP_USER = $env:SMTP_USER
$SMTP_PASS = $env:SMTP_PASS
$MAIL_FROM = if ($env:MAIL_FROM) { $env:MAIL_FROM } else { $SMTP_USER }
$MAIL_TO   = $env:MAIL_TO

Write-Log "SMTP_HOST=$SMTP_HOST PORT=$SMTP_PORT USER=$SMTP_USER"
Write-Log "MAIL_FROM=$MAIL_FROM"
Write-Log "MAIL_TO=$MAIL_TO"

# ---------- Compose MAIL (English ASCII to avoid mojibake) ----------
# 수신자 배열(콤마/세미콜론 허용)
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

# 첨부 파일 검사
$Attachments = @()
foreach ($p in @($MergedSummary, $BreakoutSummary, $BoxLineSummary)) {
  if (Test-Path $p) {
    $Attachments += (Resolve-Path $p).Path
  } else {
    Write-Log "[ATTACH][WARN] not found -> $p"
  }
}

# 메일 본문(영문, ASCII-safe)
$subject = "[Autotrade] Daily Report $DATE $TagHalf"
$bodyTxt = @"
Daily report ($DATE $TagHalf)

Attachments:
 - $(Split-Path $MergedSummary -Leaf)
 - $(Split-Path $BreakoutSummary -Leaf)
 - $(Split-Path $BoxLineSummary -Leaf)

Log file: $(Resolve-Path $EmailLog)

Notes:
 - Merged summary contains both strategies.
 - Trades are deduplicated across strategies and saved to:
   $(Split-Path $DedupTradesOut -Leaf)
"@

function Send-PlainMail {
  param(
    [Parameter(Mandatory)][string]$Subject,
    [Parameter(Mandatory)][string]$Body,
    [Parameter(Mandatory)][string[]]$ToList,
    [Parameter(Mandatory)][string]$From,
    [Parameter(Mandatory)][string]$SmtpHost,
    [Parameter(Mandatory)][int]$SmtpPort,
    [string]$User = $null,
    [string]$Pass = $null,
    [string[]]$Attachments = @(),
    [string]$LogPath = $null
  )
  $enc = [System.Text.Encoding]::ASCII   # ASCII로 안전 전송
  $msg = New-Object System.Net.Mail.MailMessage
  $msg.From = $From
  foreach ($to in $ToList) { [void]$msg.To.Add($to) }
  $msg.Subject = $Subject
  $msg.IsBodyHtml = $false

  # AlternateView (text/plain, quoted-printable)
  $alt = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($Body, $enc, "text/plain")
  $alt.TransferEncoding = [System.Net.Mime.TransferEncoding]::QuotedPrintable
  $msg.AlternateViews.Clear()
  $msg.AlternateViews.Add($alt)
  $msg.Body = $Body

  foreach ($p in $Attachments) {
    if ($p -and (Test-Path -LiteralPath $p)) {
      $att = New-Object System.Net.Mail.Attachment($p)
      $msg.Attachments.Add($att) | Out-Null
    }
  }

  $cli = New-Object System.Net.Mail.SmtpClient($SmtpHost, $SmtpPort)
  $cli.EnableSsl = $true
  if ($User -and $Pass) { $cli.Credentials = New-Object System.Net.NetworkCredential($User, $Pass) }

  try {
    $cli.Send($msg)
    if ($LogPath) { "[MAIL][OK] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Subject=$Subject" | Out-File -FilePath $LogPath -Append -Encoding UTF8 }
  } catch {
    $err = $_.Exception.Message
    if ($LogPath) { "[MAIL][ERROR] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $err" | Out-File -FilePath $LogPath -Append -Encoding UTF8 }
    throw
  } finally {
    $msg.Dispose()
    $cli.Dispose()
  }
}

try {
  Send-PlainMail `
    -Subject $subject `
    -Body $bodyTxt `
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