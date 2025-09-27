@echo off
chcp 65001 >NUL
setlocal enabledelayedexpansion

set "ROOT=D:\upbit_autotrade_starter"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "LOGDIR=%ROOT%\logs\fetch"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOG=%LOGDIR%\binance_ohlcv_append.log"

cd /d "%ROOT%"
echo [START] %DATE% %TIME% >> "%LOG%"
"%PY%" fetch_ohlcv_binance.py ^
  --symbols-file ".\configs\binance_universe.txt" ^
  --timeframe 15m ^
  --since-days 7 ^
  --append ^
  --outdir ".\data\ohlcv_binance" 1>>"%LOG%" 2>&1

set "RC=%ERRORLEVEL%"
echo [END]   %DATE% %TIME% (rc=%RC%) >> "%LOG%"
exit /b %RC%