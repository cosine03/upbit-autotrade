param(
  [string]$CsvPath = ".\logs\paper\trades_closed.csv"
)

$STD = "opened_at,symbol,event,side,level,closed_at,entry_price,exit_price,pnl,reason,fee"

if (-not (Test-Path $CsvPath)) { Write-Host "[INFO] not found: $CsvPath"; exit 0 }

$bak = "$CsvPath.bak_$(Get-Date -Format 'yyyyMMddHHmmss')"
Copy-Item $CsvPath $bak -Force
Write-Host "[BACKUP] -> $bak"

$lines = Get-Content $CsvPath
if ($lines.Count -eq 0) {
  Set-Content $CsvPath $STD
  Write-Host "[FIX] wrote header only."
  exit 0
}

if ($lines[0] -ne $STD) {
  # 헤더 교체 + 휴리스틱 재매핑
  $data = @()
  for ($i=0; $i -lt $lines.Count; $i++) {
    if ($i -eq 0 -and $lines[$i] -match "^opened_at,") { continue } # 과거 다른 헤더 버리기
    elseif ($i -eq 0 -and $lines[$i] -match "^\d{4}-\d{2}-\d{2}T") {
      # 첫 줄부터 데이터 → 그대로 둠
      $data += $lines[$i]
    }
    elseif ($i -gt 0) {
      $data += $lines[$i]
    }
  }

  # 간단한 맵핑: 11컬럼이 아니고, 뒤쪽이 reason,fee 같은 모양이면 rearrange 시도
  $fixed = @()
  foreach ($row in $data) {
    if ([string]::IsNullOrWhiteSpace($row)) { continue }
    $cells = $row.Split(',')
    if ($cells.Count -eq 11) {
      $fixed += $row
    }
    else {
      # 흔한 패턴: opened_at, symbol, event, side, level, closed_at, reason, entry, exit, pnl, fee (과거 오류)
      if ($cells.Count -ge 11 -and $cells[6] -notmatch "^\d+(\.\d+)?$") {
        $reordered = @($cells[0..5] + @($cells[7],$cells[8],$cells[9]) + @($cells[6],$cells[10]))
        $fixed += ($reordered -join ',')
      } else {
        $fixed += $row
      }
    }
  }

  Set-Content $CsvPath $STD
  Add-Content $CsvPath ($fixed -join "`n")
  Write-Host "[FIX] header normalized."
} else {
  Write-Host "[OK] already normalized."
}