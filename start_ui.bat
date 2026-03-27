@echo off
REM KGoT Streamlit UI Launcher
REM Starts the Streamlit app and opens browser

SETLOCAL EnableDelayedExpansion

SET "SCRIPT_DIR=%~dp0"
SET "PORT=8765"
SET "URL=http://localhost:%PORT%"

echo.
echo ========================================
echo   KGoT Console - Streamlit UI
echo ========================================
echo.
echo Starting Streamlit server on port %PORT%...
echo URL: %URL%
echo.

REM Check if venv exists
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    echo Using virtual environment: %SCRIPT_DIR%.venv
    SET "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    echo Using system Python
    SET "PYTHON_EXE=python"
)

REM Open browser after a short delay
echo Opening browser in 3 seconds...
start "" cmd /c "timeout /t 3 /nobreak >nul && start %URL%"

REM Start Streamlit
echo.
echo Starting Streamlit...
echo Press Ctrl+C to stop the server
echo.

"%PYTHON_EXE%" -m streamlit run "%SCRIPT_DIR%streamlit_app.py" --server.port %PORT% --server.headless true

ENDLOCAL