@echo off
REM ============================================================================
REM KERTAS Paleographer - Docker Quick Start Script (Windows)
REM ============================================================================

echo ╔═══════════════════════════════════════════════════════════════╗
echo ║     🐳 KERTAS Paleographer - Docker Deployment              ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed!
    pause
    exit /b 1
)

echo ✓ Docker is installed
echo ✓ Docker Compose is installed
echo.

REM Menu
echo Select deployment option:
echo.
echo   1) 🚀 Start Web Application (Recommended)
echo   2) 💻 Start CLI Interface
echo   3) 🔨 Build Docker Image Only
echo   4) 🛑 Stop All Services
echo   5) 📊 View Logs
echo   6) 🔄 Rebuild and Restart
echo   7) 🗑️  Clean Up (Remove containers ^& images)
echo.

set /p choice="Enter choice [1-7]: "

if "%choice%"=="1" (
    echo Starting Web Application...
    docker-compose up -d app
    echo.
    echo ✅ Application started successfully!
    echo Access the web interface at: http://localhost:8501
    echo.
    echo View logs with: docker-compose logs -f app
    start http://localhost:8501
)

if "%choice%"=="2" (
    echo Starting CLI Interface...
    docker-compose --profile cli run --rm cli
)

if "%choice%"=="3" (
    echo Building Docker image...
    docker build -t kertas-paleographer:latest .
    echo ✅ Build complete!
)

if "%choice%"=="4" (
    echo Stopping all services...
    docker-compose down
    echo ✅ All services stopped
)

if "%choice%"=="5" (
    echo Showing logs (Ctrl+C to exit)...
    docker-compose logs -f
)

if "%choice%"=="6" (
    echo Rebuilding and restarting...
    docker-compose down
    docker-compose up -d --build
    echo ✅ Application rebuilt and restarted!
    echo Access at: http://localhost:8501
    start http://localhost:8501
)

if "%choice%"=="7" (
    set /p confirm="⚠️  This will remove all containers and images. Are you sure? (y/N): "
    if /i "%confirm%"=="y" (
        echo Cleaning up...
        docker-compose down -v
        docker rmi kertas-paleographer:latest 2>nul
        echo ✅ Cleanup complete
    ) else (
        echo Cleanup cancelled
    )
)

echo.
echo For more information, see: DOCKER.md
pause

