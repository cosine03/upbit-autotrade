# upbit-autotrade
Upbit Autotrade Repositary I Use with Charty Harry

추천 실행 방법 (PowerShell 예시)

1) TV(alt) + 업비트(alt)를 동시에 다른 PS 창에서

창 A (TV ALT):


$logA = ".\logs\tv_mp_alt_$(Get-Date -Format yyyyMMdd_HHmm).log"
python backtest_tv_entry_close_mp.py .\logs\signals_tv.csv --group alt --expiry 4h,8h --procs 20 `
  | Tee-Object -FilePath $logA -Append

창 B (UPBIT ALT):


$logB = ".\logs\upbit_mp_alt_$(Get-Date -Format yyyyMMdd_HHmm).log"
python backtest_upbit_multi_strategies_mp.py .\logs\signals_upbit.csv --group alt --horizon 24h --procs 20 `
  | Tee-Object -FilePath $logB -Append

> 메모: --procs는 코어 여유에 맞춰 조절(보통 CPU코어-1). 두 잡을 동시에 돌릴 땐 각 4~6 정도로 낮추면 안정적.

2) majors 따로 돌릴 때

창 C (TV MAJOR):


$logC = ".\logs\tv_mp_major_$(Get-Date -Format yyyyMMdd_HHmm).log"
python backtest_tv_entry_close_mp.py .\logs\signals_tv.csv --group major --expiry 4h,8h --procs 4 `
  | Tee-Object -FilePath $logC -Append

창 D (UPBIT MAJOR):


$logD = ".\logs\upbit_mp_major_$(Get-Date -Format yyyyMMdd_HHmm).log"
python backtest_upbit_multi_strategies_mp.py .\logs\signals_upbit.csv --group major --horizon 24h --procs 4 `
  | Tee-Object -FilePath $logD -Append

속도/안정 팁

동일 스크립트 중복 실행 금지: 같은 스크립트를 두 창에서 동시에 돌리면 심볼 캐시 CSV를 동시에 쓰려다 경합 날 수 있어. (TV vs UPBIT는 폴더가 달라서 OK)

캐시 워밍업(선택): 처음 한 번 알트/메이저를 분리해서 짧게 돌려두면 캐시가 채워져 후속 실행이 빨라짐.

python backtest_tv_entry_close_mp.py .\logs\signals_tv.csv --group alt --expiry 4h --procs 8

심볼 부분 실행: 특정 코인만 빠르게 검증

python backtest_upbit_multi_strategies_mp.py .\logs\signals_upbit.csv --symbols KRW-SOL,KRW-XRP --procs 2

로그 분리: 위처럼 Tee-Object로 로그 파일을 각 잡별로 남겨두면 진행상황/에러 추적이 쉬워.