@echo off
chcp 936 >nul 2>&1
title PA Agent - AI K线分析助手
cd /d "%~dp0"
rem 自动探测 Python 3.12（优先），找不到就用 PATH 里的 python
set "PYEXE="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYEXE for /f "delims=" %%i in ('where python 2^>nul') do (set "PYEXE=%%i" & goto :run)
:run
"%PYEXE%" run.py
if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出，请查看上方错误信息。
    pause
)
