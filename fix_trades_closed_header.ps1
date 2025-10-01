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

# 표준(신) 헤더 11열
$stdHeader = @(
  'opened_at','symbol','event','side','level',
  'closed_at','entry_price','exit_price','pnl','reason','fee'
)

# 옛(구) 헤더 8열
$oldHeader = @(
  'opened_at','symbol','event','side','level',
  'closed_at','reason','fee'
)

# 파일 읽기 (첫 줄=헤더는 버리고, 나머지 라인 직접 처리)
$lines = Get-Content -LiteralPath $Path -Encoding UTF8
if ($lines.Count -lt 2) {
  Set-Content -LiteralPath $Path -Encoding UTF8 -Value ($stdHeader -join ',')
  Write-Host "ℹ️  데이터가 없어 표준 헤더만 기록했습니다."
  exit 0
}
$dataLines = $lines | Select-Object -Skip 1

# CSV 안전 분리기(간단 버전: 큰따옴표 없는 전제)
function SplitCsvSimple([string]$line) {
  # 큰따옴표 포함 필드가 없다면 단순 split으로 충분
  return $line.Split(',')
}

# 새 출력 버퍼
$out = New-Object System.Collections.Generic.List[string]
$out.Add(($stdHeader -join ','))

[int]$migrated8 = 0
[int]$kept11 = 0
[int]$skipped = 0

foreach ($raw in $dataLines) {
  if ([string]::IsNullOrWhiteSpace($raw)) { continue }
  $cols = SplitCsvSimple $raw.Trim()

  switch ($cols.Count) {
    8 {
      # 8열(구) → 11열(신)으로 승격
      $obj = [ordered]@{}
      $obj.opened_at = $cols[0]
      $obj.symbol    = $cols[1]
      $obj.event     = $cols[2]
      $obj.side      = $cols[3]
      $obj.level     = $cols[4]
      $obj.closed_at = $cols[5]
      # 신 필드 채우기
      $obj.entry_price = ''
      $obj.exit_price  = ''
      $obj.pnl         = ''
      $obj.reason      = $cols[6]
      $obj.fee         = $cols[7]

      $out.Add(($stdHeader | ForEach-Object { $obj[$_] }) -join ',')
      $migrated8++
    }
    11 {
      # 이미 신 스키마 → 순서만 표준화해서 재기록
      $obj = [ordered]@{}
      for ($i=0; $i -lt 11; $i++) {
        $obj[$stdHeader[$i]] = $cols[$i]
      }
      $out.Add(($stdHeader | ForEach-Object { $obj[$_] }) -join ',')
      $kept11++
    }
    default {
      # 예상 밖 라인은 스킵(백업본에서 확인 가능)
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

# Sanity check
Import-Csv -LiteralPath $Path | Select-Object -First 3 | Format-List