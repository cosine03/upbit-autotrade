# 0) 경로/폴더 잡기
$DATE = Get-Date -Format "yyyy-MM-dd"
$TAG  = "${DATE}_SUPPORT_TEST"
$OUT  = ".\logs\daily\$TAG"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# 1) 소스 지정 (support 라벨 있는 원본)
$SRC = ".\logs\signals_tv.csv"   # 네 파일 경로로 맞춰둔 상태

# 2) 라벨링(사이드/터치회수 등) - 안전하게 항상 거쳐가자
python -u .\label_events_side.py $SRC `
  --out "$OUT\signals_labeled.csv"

# 3) support 전용 필터
#    distance_pct가 없으면 자동으로 스킵하고 저장만 함(정상 동작)
python -u .\filter_support_events.py "$OUT\signals_labeled.csv" `
  --out "$OUT\signals_support_only.csv" `
  --dist-max 0.000188 `
  --include-substr "support,price_in_box,bounce"
#  --require-same-level   # 필요하면 나중에 켜기

# 4) 결과 확인 (행수만 체크)
(Import-Csv "$OUT\signals_support_only.csv" | Measure-Object).Count