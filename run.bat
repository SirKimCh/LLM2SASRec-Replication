@echo off
rem One command for any Windows machine. Finds uv (or downloads the standalone binary into .tools\), keeps every cache
rem inside the project folder, builds .venv exactly from uv.lock, then 99_master_run.py checks/downloads the rest and runs.
rem   run.bat                      full schedule (tiers A, B, C), resumes after any interruption
rem   run.bat --tier A             stop after tier A (the core comparisons)
rem   run.bat --estimate           time estimate only
rem   run.bat --smoke              end-to-end self-test on small samples (1 seed, 1 epoch) into Result_smoke\
rem   run.bat --budget-hours 8     time-guard for a run on the development laptop (no new unit after 80 %%)
rem While it runs: a TIEN DO line (bar, units, hours left) after every unit, also in the window title. On any failure the
rem window turns red, beeps and waits, so an unattended run cannot fail silently.
rem When the results have been copied away: clean.bat removes everything this file installed and keeps code + Result\.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "TF_CPP_MIN_LOG_LEVEL=2"
set "PYTHONHASHSEED=0"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

rem cache root: the shared workspace cache when src\ sits inside the research workspace (has 0_Init\ two levels up), else src\.cache
set "CACHE=%~dp0.cache"
if exist "%~dp0..\..\0_Init\" if exist "E:\NCKH\.cache\" set "CACHE=E:\NCKH\.cache"
set "UV_CACHE_DIR=%CACHE%\uv"
set "UV_PYTHON_INSTALL_DIR=%CACHE%\uv-python"
set "PIP_CACHE_DIR=%CACHE%\pip"
set "HF_HOME=%CACHE%\huggingface"
set "FASTEMBED_CACHE_PATH=%CACHE%\fastembed"
set "MPLCONFIGDIR=%CACHE%\matplotlib"
rem a Python that uv has to download stays inside the cache above: no entry in the Windows registry
set "UV_PYTHON_INSTALL_REGISTRY=0"

where uv >nul 2>nul
if not errorlevel 1 goto run
if exist "%~dp0.tools\uv.exe" goto localuv
echo [run] uv not found - downloading the standalone binary into .tools\ ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $d=Join-Path (Get-Location) '.tools'; New-Item -ItemType Directory -Force $d | Out-Null; Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile (Join-Path $d 'uv.zip'); Expand-Archive (Join-Path $d 'uv.zip') $d -Force; Remove-Item (Join-Path $d 'uv.zip')"
if errorlevel 1 goto nouv
:localuv
set "PATH=%~dp0.tools;%PATH%"
:run
rem --frozen: install exactly what uv.lock says; another uv version must not re-resolve it
uv run --frozen python 99_master_run.py %*
set "RC=%errorlevel%"
if "%RC%"=="0" goto done
if "%RC%"=="75" goto budget
goto failed
:nouv
echo [run] could not download uv. Install it from https://docs.astral.sh/uv/ and run this file again.
set "RC=1"
:failed
color 4F
title SEQREC - DA DUNG VI LOI (ma thoat %RC%)
echo.
echo ======================================================================
echo   DA DUNG VI LOI  (ma thoat %RC%)
echo   Chi tiet: dong CRASH cuoi cung trong Result\master_log.txt
echo            (smoke: Result_smoke\master_log.txt)
echo   Sua xong thi chay lai dung lenh nay - no chay tiep tu cho da dung.
echo ======================================================================
powershell -NoProfile -Command "[console]::beep(880,400); [console]::beep(660,600)" >nul 2>nul
pause
color
exit /b %RC%
:budget
echo [run] Het ngan sach gio may (--budget-hours): da phan tich tren phan da chay. Chay tiep: bo --budget-hours hoac dat lon hon.
:done
rem keep a double-clicked window open so the last lines stay readable (returns at once when there is no keyboard)
pause
exit /b %RC%
