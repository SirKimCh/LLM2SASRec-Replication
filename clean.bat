@echo off
rem Undoes what run.bat put on this machine and keeps the code and Result\. Removed: the virtual environment, raw and
rem derived data, the project-local caches and binaries, smoke leftovers and the machine reports written by 00.
rem Nothing that existed before run.bat is touched: when this folder sits inside the research workspace, the workspace's
rem shared .cache is left alone. The next run.bat downloads and builds everything again; finished work is still
rem recognised from Result\.
rem   clean.bat          lists what will be deleted and asks for the word XOA
rem   clean.bat --yes    no question
setlocal
chcp 65001 >nul
cd /d "%~dp0" || exit /b 1
if not exist "99_master_run.py" (echo [clean] Day khong phai thu muc src cua du an - khong lam gi. & exit /b 1)
set "RC=0"
set "DIRS=.venv DataOG dataset dataset_smoke Result_smoke .cache .tools .pytest_cache"
set "FILES=hardware_profile.json resource_plan.json"

rem a run in progress (or a test session) uses .venv: deleting under it would kill the run half-way
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith((Join-Path (Get-Location).Path '.venv\'), 'OrdinalIgnoreCase') }) { exit 1 }"
if errorlevel 1 goto busy

echo.
echo Thu muc: %CD%
echo Se XOA - giu lai code va Result\ :
for %%D in (%DIRS%) do if exist "%%D\" echo   %%D\
for %%F in (%FILES%) do if exist "%%F" echo   %%F
echo   __pycache__\ o moi cap
echo Lan chay run.bat sau se tai va cai lai tu dau.
if not exist "Result\ThucNghiem_Log.csv" echo !! Result\ chua co ket qua nao.
echo.
if /i "%~1"=="--yes" goto go
set "OK="
set /p "OK=Go XOA roi Enter de xac nhan (go khac = huy): "
if /i not "%OK%"=="XOA" goto cancel

:go
for %%D in (%DIRS%) do call :rmd "%%D"
for /d /r %%P in (__pycache__) do if exist "%%P\" rd /s /q "%%P"
for %%F in (%FILES%) do if exist "%%F" del /f /q "%%F"
echo.
if "%RC%"=="0" (echo [clean] Xong. Con lai trong thu muc:) else (echo [clean] CHUA SACH HET - xem cac dong tren. Con lai:)
dir /b
goto end

:rmd
if not exist "%~1\" exit /b 0
rd /s /q "%~1" 2>nul
rem paths longer than 260 characters (package caches) need the \\?\ form
if exist "%~1\" rd /s /q "\\?\%CD%\%~1" 2>nul
if exist "%~1\" (echo [clean] KHONG XOA DUOC %~1\ - co tien trinh dang dung no? & set "RC=1")
exit /b 0

:busy
echo [clean] Dang co tien trinh chay bang .venv cua thu muc nay - run.bat dang chay? Dong no roi chay lai clean.bat.
set "RC=1"
goto end

:cancel
echo [clean] Da huy - khong xoa gi.
set "RC=1"

:end
rem keep a double-clicked window open (returns at once when there is no keyboard)
pause
exit /b %RC%
