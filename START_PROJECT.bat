@echo off
REM Citizen Grievance Analysis System - Windows Startup Script
REM This script starts FastAPI and Streamlit services in separate windows

setlocal enabledelayedexpansion

title Citizen Grievance Analysis System

REM Colors setup (using cls for visual separation)
echo.
echo ================================================================================
echo   CITIZEN GRIEVANCE ANALYSIS SYSTEM - WINDOWS LAUNCHER
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python is installed
echo.

REM Change to project directory
cd /d "%~dp0"
echo [INFO] Working directory: %CD%

REM Check requirements
echo [CHECK] Verifying dependencies...
python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [ACTION] Installing requirements...
    call python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements
        pause
        exit /b 1
    )
)
echo [SUCCESS] Dependencies verified
echo.

REM Check/download dataset
if exist "datasets\*.csv" (
    echo [SUCCESS] Dataset exists
) else (
    echo [ACTION] Downloading dataset...
    call python download_dataset.py
    if errorlevel 1 (
        echo [WARNING] Dataset download failed - will use synthetic data
    ) else (
        echo [SUCCESS] Dataset downloaded
    )
)
echo.

REM Check/train models
if exist "trained_models\sentiment_model.joblib" (
    echo [SUCCESS] Models are trained
) else (
    echo [ACTION] Training models (this may take 2-5 minutes)...
    echo Starting Jupyter conversion...
    python -m jupyter nbconvert ^
        --to notebook ^
        --execute ^
        --ExecutePreprocessor.timeout=3600 ^
        "WEEK3_IMPROVED_ORIGINAL_DATASET.ipynb"
    
    if errorlevel 1 (
        echo [WARNING] Model training failed - check WEEK3_IMPROVED_ORIGINAL_DATASET.ipynb
    ) else (
        echo [SUCCESS] Models trained
    )
)
echo.

REM Start FastAPI in new window
echo [ACTION] Starting FastAPI Server...
start "FastAPI - Citizen Grievance API" cmd /k python -m uvicorn api.main:app --reload --port 8000

REM Wait for FastAPI to start
timeout /t 2 /nobreak

REM Start Streamlit in new window
echo [ACTION] Starting Streamlit UI...
start "Streamlit - Citizen Grievance UI" cmd /k python -m streamlit run streamlit_app.py --server.port 8501

REM Display information
echo.
echo ================================================================================
echo   SERVICES STARTING
echo ================================================================================
echo.
echo FastAPI Server:
echo   URL: http://localhost:8000
echo   Documentation: http://localhost:8000/docs
echo   Status: Opening in new window...
echo.
echo Streamlit Interface:
echo   URL: http://localhost:8501
echo   Status: Opening in new window...
echo.
echo ================================================================================
echo.
echo [INFO] Waiting for services to initialize (30 seconds)...
timeout /t 5 /nobreak

echo.
echo [SUCCESS] System is ready!
echo.
echo Next Steps:
echo   1. Open your browser to http://localhost:8501
echo   2. Try the three modes: Single Prediction, Batch Processing, System Info
echo   3. Check API documentation at http://localhost:8000/docs
echo.
echo [TIP] You can close either window independently:
echo   - Close FastAPI window to stop the API
echo   - Close Streamlit window to stop the UI
echo   - Close this window to view service logs
echo.
pause
