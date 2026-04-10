@echo off
REM ============================================================
REM NYC 311 Citizen Grievance Analysis - Windows Startup Script
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  NYC 311 CITIZEN GRIEVANCE ANALYSIS - PROJECT STARTUP
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not installed
    pause
    exit /b 1
)

echo [OK] pip found

REM Set project directory
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo.
echo [INFO] Project directory: %PROJECT_DIR%
echo.

REM Install requirements
echo [INFO] Checking dependencies...
python -m pip install --upgrade pip -q >nul 2>&1

if not exist requirements.txt (
    echo [WARN] requirements.txt not found, skipping dependency installation
) else (
    echo [INFO] Installing requirements from requirements.txt...
    pip install -r requirements.txt -q
    if errorlevel 1 (
        echo [WARN] Some dependencies failed to install, but continuing anyway...
    )
)

echo.
echo ============================================================
echo  STARTING PROJECT WORKFLOW
echo ============================================================
echo.

REM Parse command line arguments
set SKIP_DOWNLOAD=
set SKIP_TRAIN=
set START_SERVICES=
set API_PORT=8000
set UI_PORT=8501

:parse_args
if "%1"=="" goto run_project
if "%1"=="--skip-download" set SKIP_DOWNLOAD=True
if "%1"=="--skip-train" set SKIP_TRAIN=True
if "%1"=="--start-services" set START_SERVICES=True
if "%1"=="--api-port" set API_PORT=%2
if "%1"=="--ui-port" set UI_PORT=%2
shift
goto parse_args

:run_project
echo [INFO] Running project with options:
if defined SKIP_DOWNLOAD echo   - Skip Download: YES
if defined SKIP_TRAIN echo   - Skip Training: YES
if defined START_SERVICES echo   - Start Services: YES
echo   - API Port: %API_PORT%
echo   - UI Port: %UI_PORT%
echo.

REM Run the main project script
python run_project.py ^
    %if defined SKIP_DOWNLOAD% --skip-download %else% % ^
    %if defined SKIP_TRAIN% --skip-train %else% % ^
    %if defined START_SERVICES% --start-services %else% %

if errorlevel 1 (
    echo.
    echo [ERROR] Project execution failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  PROJECT COMPLETED SUCCESSFULLY
echo ============================================================
echo.
echo [OK] Sentiment analysis models trained on complete dataset
echo [OK] Deployment artifacts prepared
echo.
echo [INFO] Next steps:
echo   1. Check trained models in: trained_models/
echo   2. Review outputs in: outputs/
echo   3. Check logs in: logs/project.log
echo.

pause

goto EOF

REM This section below is now obsolete - moved to run_project.py

:EOF

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
