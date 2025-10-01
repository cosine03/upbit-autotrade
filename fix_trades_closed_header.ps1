# trades_closed.csv 헤더를 표준 헤더로 교체하는 스크립트
# 기존 데이터는 유지하면서 헤더만 교체

$path = ".\logs\paper\trades_closed.csv"

if (Test-Path $path) {
    $rows = Get-Content $path
    if ($rows.Count -gt 0) {
        # 올바른 헤더 정의 (paper_trader.py 기준)
        $correct = "opened_at,symbol,event,side,level,closed_at,entry_price,exit_price,pnl,reason,fee"

        # 첫 줄이 올바르지 않으면 교체
        if ($rows[0] -ne $correct) {
            Write-Host "기존 헤더 수정 중..."
            $rows[0] = $correct
            $rows | Set-Content -Path $path -Encoding UTF8
            Write-Host "헤더 교체 완료 ✅"
        }
        else {
            Write-Host "이미 올바른 헤더입니다 ✅"
        }
    }
    else {
        Write-Host "파일이 비어 있습니다."
    }
}
else {
    Write-Host "파일이 존재하지 않습니다: $path"
}