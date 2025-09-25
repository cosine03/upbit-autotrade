# 0) 원본 파일 지정 (네가 올린 그 파일 경로에 맞춰)
$SRC = ".\logs\signals\signals_2025-09-25_AM.csv"   # 네 환경파일로 교체
$OUT = ".\logs\daily\2025-09-25_AM"

# 1) side 라벨이 이미 있으므로, 라벨링은 스킵 가능 (없으면 label_events_side.py 사용)
# python -u .\label_events_side.py $SRC --out "$OUT\signals_labeled.csv"

# 2) support only + price_in_box만 우선 추출
python -u .\filter_support_events.py $SRC `
  --out "$OUT\signals_support_only.csv" `
  --dist-max 0.000188 `
  --include-substr "support,price_in_box,bounce"    # price_in_box 포함시켜둠
# (--require-same-level 옵션은 나중에 켜서 더 타이트하게 테스트)

# 3) 백테스트 (네 베이스라인과 동일 파라미터)
python -u .\backtest_tv_events_mp.py "$OUT\signals_support_only.csv" `
  --timeframe 15m --expiries 0.5h,1h,2h `
  --tp 1.75 --sl 0.7 --fee 0.001 `
  --dist-max 9 `
  --procs 24 `
  --ohlcv-roots ".;.\data;.\data\ohlcv;.\ohlcv;.\logs;.\logs\ohlcv" `
  --ohlcv-patterns "data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv" `
  --assume-ohlcv-tz UTC `
  --outdir "$OUT\bt_support_only"