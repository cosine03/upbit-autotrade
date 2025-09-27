
# Quickstart: Binance signals pipeline (first step)

# 1) Generate Binance-only signals from your raw TradingView sink (signals_tv.csv)
#    - Produces:
#        .\logs\signals_binance_raw.csv          (BTCUSDT etc. as-is)
#        .\logs\signals_binance_as_upbit.csv     (BTCUSDT -> KRW-BTC mapping)
#
# PowerShell (from project root):
.\.venv\Scripts\python.exe .\extract_binance_signals.py ^
  --src .\logs\signals_tv.csv ^
  --out-raw .\logs\signals_binance_raw.csv ^
  --out-upbit .\logs\signals_binance_as_upbit.csv

# 2) Backtest (Upbit price) using mapped signals
.\.venv\Scripts\python.exe .\backtest_tv_events_mp.py .\logs\signals_binance_as_upbit.csv ^
  --timeframe 15m --expiries 0.5h,1h,2h ^
  --tp 1.75 --sl 0.7 --fee 0.001 ^
  --dist-max 0.00025 ^
  --procs 24 ^
  --ohlcv-roots ".;.\data;.\data\ohlcv;.\ohlcv;.\logs;.\logs\ohlcv" ^
  --ohlcv-patterns "data/ohlcv/{symbol}-{timeframe}.csv;data/ohlcv/{symbol}_{timeframe}.csv;{symbol}-{timeframe}.csv;{symbol}_{timeframe}.csv" ^
  --assume-ohlcv-tz UTC ^
  --outdir .\logs\bt_binance_alarm_upbit_price

# 3) (Later) Backtest (Binance price) using raw signals
#    - when Binance OHLCV is ready and filenames match (e.g., BTCUSDT-15m.csv)
.\.venv\Scripts\python.exe .\backtest_tv_events_mp.py .\logs\signals_binance_raw.csv ^
  --timeframe 15m --expiries 0.5h,1h,2h ^
  --tp 1.75 --sl 0.7 --fee 0.001 ^
  --dist-max 0.00025 ^
  --procs 24 ^
  --ohlcv-roots ".;.\data;.\data\ohlcv_binance;.\ohlcv" ^
  --ohlcv-patterns "data/ohlcv_binance/{symbol}-{timeframe}.csv;{symbol}-{timeframe}.csv" ^
  --assume-ohlcv-tz UTC ^
  --outdir .\logs\bt_binance_alarm_binance_price
