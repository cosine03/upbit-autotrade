Param(
  [string]$Path = ".\logs\paper\trades_closed.csv",
  [string]$BackupDir = ".\logs\paper\_backup"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Path)) {
  Write-Error "파일을 찾을 수 없음: $Path"
  exit 1
}

# 백업
if (-not (Test-Path $BackupDir)) { New-Item -ItemType Directory -Path $BackupDir | Out-Null }
$stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$backupPath = Join-Path $BackupDir ("trades_closed_" + $stamp + ".csv")
Copy-Item $Path $backupPath -Force
Write-Host "🗂️  백업 생성: $backupPath"

# 표준 헤더(11열, PnL 포함)
$stdHeader = "opened_at,symbol,event,side,level,closed_at,entry_price,exit_price,pnl,reason,fee"

# 파일 전체 로드
$lines = Get-Content -LiteralPath $Path -Encoding UTF8

if ($lines.Count -lt 2) {
  # 헤더만 있거나 비어있을 때 → 표준 헤더만 강제
  Set-Content -LiteralPath $Path -Value $stdHeader -Encoding UTF8
  Write-Host "ℹ️  데이터가 없어서 표준 헤더만 기록했습니다."
  exit 0
}

# 현재 헤더 감지
$currentHeader = $lines[0].Trim()
$dataLines = $lines | Select-Object -Skip 1

# 각 라인의 필드 수를 안전하게 계산 (trades_closed.csv에는 따옴표/콤마 포함 필드가 없다는 전제)
function SplitSafe([string]$line) {
  # 필요 시 더 강한 CSV 파서로 교체 가능
  return $line.Split(",")
}

# 새 출력용 버퍼 (표준 헤더부터)
$out = New-Object System.Collections.Generic.List[string]
$out.Add($stdHeader)

[int]$migrated8 = 0
[int]$kept11 = 0
[int]$skipped = 0

foreach ($raw in $dataLines) {
  $rawTrim = $raw.Trim()
  if ([string]::IsNullOrWhiteSpace($rawTrim)) { continue }

  $cols = SplitSafe $rawTrim

  switch ($cols.Count) {
    8 {
      # 구(8열) → 신(11열)로 승격
      # 0 opened_at
      # 1 symbol
      # 2 event
      # 3 side
      # 4 level
      # 5 closed_at
      # 6 reason
      # 7 fee
      $opened_at = $cols[0]
      $symbol    = $cols[1]
      $event     = $cols[2]
      $side      = $cols[3]
      $level     = $cols[4]
      $closed_at = $cols[5]
      $reason    = $cols[6]
      $fee       = $cols[7]

      $entry_price = ""
      $exit_price  = ""
      $pnl         = ""

      $newline = ($opened_at,$symbol,$event,$side,$level,$closed_at,$entry_price,$exit_price,$pnl,$reason,$fee) -join ","
      $out.Add($newline)
      $migrated8++
    }
    11 {
      # 이미 신 스키마. 그대로 기록 (혹시 순서가 다르면 매핑해서 재정렬)
      $opened_at   = $cols[0]
      $symbol      = $cols[1]
      $event       = $cols[2]
      $side        = $cols[3]
      $level       = $cols[4]
      $closed_at   = $cols[5]
      $entry_price = $cols[6]
      $exit_price  = $cols[7]
      $pnl         = $cols[8]
      $reason      = $cols[9]
      $fee         = $cols[10]

      $newline = ($opened_at,$symbol,$event,$side,$level,$closed_at,$entry_price,$exit_price,$pnl,$reason,$fee) -join ","
      $out.Add($newline)
      $kept11++
    }
    default {
      # 예상치 못한 라인 → 스킵(백업본에서 확인)
      $skipped++
    }
  }
}

# 덮어쓰기
$out | Set-Content -LiteralPath $Path -Encoding UTF8
Write-Host "✅ 정규화 완료: $Path"
Write-Host "   - 8열 → 11열 승격: $migrated8"
Write-Host "   - 11열 유지:        $kept11"
if ($skipped -gt 0) {
  Write-Warning "   - 스킵된 라인:      $skipped (백업본에서 확인 요망)"
}

# Sanity check: 첫 행 출력
Import-Csv $Path | Select-Object -First 1 | Format-List