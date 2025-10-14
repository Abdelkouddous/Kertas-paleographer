@echo off
REM Quick Launch Script for ML Classification Web UI (Windows)

echo.
echo 🚀 Launching ML Classification Web UI...
echo.
echo 📍 Location: http://localhost:8501
echo ⚡ Press Ctrl+C to stop the server
echo.
echo 🎨 Opening in browser...
echo.

REM Navigate to script directory
cd /d "%~dp0"

REM Launch Streamlit
streamlit run app.py --server.headless false

echo.
echo ✅ Server stopped. Goodbye!
pause

