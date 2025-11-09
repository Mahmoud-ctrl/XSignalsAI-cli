@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title XSignals AI - Crypto Analysis Toolkit
color 0A
cls

echo.
echo ██╗  ██╗███████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗     ███████╗ █████╗ ██╗
echo ╚██╗██╔╝██╔════╝██║██╔════╝ ████╗  ██║██╔══██╗██║     ██╔════╝██╔══██╗██║
echo  ╚███╔╝ ███████╗██║██║  ███╗██╔██╗ ██║███████║██║     ███████╗███████║██║
echo  ██╔██╗ ╚════██║██║██║   ██║██║╚██╗██║██╔══██║██║     ╚════██║██╔══██║██║
echo ██╔╝ ██╗███████║██║╚██████╔╝██║ ╚████║██║  ██║███████╗███████║██║  ██║██║
echo ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝                                                      
echo.
echo ───────────────────────────────────────────────────────────────
echo           Automated Installation Environment Setup             
echo ───────────────────────────────────────────────────────────────
echo.
timeout /t 1 >nul

:: Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Create virtual environment if missing
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [INFO] Virtual environment created.
    echo.
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Check for dependencies
if not exist "venv\Lib\site-packages\pandas\" (
    echo [INFO] Installing dependencies...
    pip install -r src\requirements.txt --quiet
    echo [INFO] Dependencies installed successfully.
    echo.
)

:MENU
cls
echo.
echo ██╗  ██╗███████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗     ███████╗ █████╗ ██╗
echo ╚██╗██╔╝██╔════╝██║██╔════╝ ████╗  ██║██╔══██╗██║     ██╔════╝██╔══██╗██║
echo  ╚███╔╝ ███████╗██║██║  ███╗██╔██╗ ██║███████║██║     ███████╗███████║██║
echo  ██╔██╗ ╚════██║██║██║   ██║██║╚██╗██║██╔══██║██║     ╚════██║██╔══██║██║
echo ██╔╝ ██╗███████║██║╚██████╔╝██║ ╚████║██║  ██║███████╗███████║██║  ██║██║
echo ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝                                                      
echo.
echo ───────────────────────────────────────────────────────────────
echo           XSignals AI - Crypto Analysis Toolkit             
echo ───────────────────────────────────────────────────────────────
echo.
timeout /t 1 >nul
echo.
echo   [1] Analyze a single coin
echo   [2] Scan for opportunities
echo   [3] Multi-timeframe analysis
echo   [4] Analyze specific pairs
echo   [5] Exit
echo.
set /p choice="Select an option (1-5): "

if "%choice%"=="1" (
    set /p symbol="Enter symbol (e.g. BTCUSDT): "
    python src\main.py --symbol %symbol%
    pause
    goto MENU
)

if "%choice%"=="2" (
    python src\main.py --scan
    pause
    goto MENU
)

if "%choice%"=="3" (
    set /p symbol="Enter symbol (e.g. ETHUSDT): "
    set /p timeframe="Enter timeframe(s) separated by spaces (e.g. 1h 4h 1d): "
    echo DEBUG: symbol=!symbol!
    echo DEBUG: timeframe=!timeframe!
    python src\main.py --symbol "!symbol!" --timeframes !timeframe!
    pause
    goto MENU
)

if "%choice%"=="4" (
    set /p pairs="Enter pairs separated by spaces (e.g. BTCUSDT ETHUSDT SOLUSDT): "
    python src\main.py --scan --pairs !pairs!
    pause
    goto MENU
)

if "%choice%"=="5" (
    echo Exiting XSignals AI...
    timeout /t 1 >nul
    exit /b
)

echo Invalid selection. Please try again.
pause
goto MENU
