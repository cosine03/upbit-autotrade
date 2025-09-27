@echo off
REM ===== Binance OHLCV append fetch (15m) =====
setlocal enabledelayedexpansion

REM --- paths ---
set "ROOT=D:\upbit_autotrade_starter"
set "VENV=%ROOT%\.venv\Scripts"
set "PY=%VENV%\python.exe"
set "LOGDIR=%ROOT%\logs\fetch"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM --- log file (고정 로그 + 누적 append) ---
set "LOG=%LOGDIR%\binance_ohlcv_append.log"

REM --- run ---
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