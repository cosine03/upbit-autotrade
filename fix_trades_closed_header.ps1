# fix_trades_closed_header.ps1
# trades_closed.csv 헤더를 최신 포맷으로 교체하는 스크립트
# 기존 데이터는 보존하고, 컬럼 순서만 수정

$path = ".\logs\paper\trades_closed.csv"
$tmp  = ".\logs\paper\trades_closed_tmp.csv"

if (-not (Test-Path $path)) {
    Write-Host "[ERROR] 파일을 찾을 수 없습니다: $path"
    exit 1
}

Write-Host "[INFO] 기존 파일 확인: $path"
Write-Host "[INFO] 임시 파일 생성: $tmp"

# 새 헤더 정의 (최신 스펙)
$newHeader = "opened_at,symbol,event,side,level,closed_at,entry_price,exit_price,pnl,reason,fee"

# 원본 읽기
$lines = Get-Content $path

if ($lines.Count -eq 0) {
    Write-Host "[WARN] 파일이 비어있습니다."
    exit 0
}

# 첫 줄(기존 헤더) 교체
$lines[0] = $newHeader

# 새 파일 쓰기
$lines | Set-Content $tmp -Encoding UTF8

# 기존 파일 백업 후 교체
$backup = "$path.bak_$(Get-Date -Format 'yyyyMMddHHmmss')"
Move-Item $path $backup
Move-Item $tmp $path

Write-Host "[INFO] 헤더 교체 완료"
Write-Host "[INFO] 백업 파일: $backup"