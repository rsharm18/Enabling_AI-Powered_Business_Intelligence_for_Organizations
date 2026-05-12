@echo off
REM AI-Powered Business Intelligence Platform - Startup Script

if "%1"=="" goto help
if "%1"=="web" goto web
if "%1"=="csv" goto csv
if "%1"=="process" goto process
if "%1"=="stop" goto stop
if "%1"=="kill-port" goto kill-port
if "%1"=="start-db" goto start-db
if "%1"=="stop-db" goto stop-db
if "%1"=="clean" goto clean
goto help

:help
echo AI-Powered Business Intelligence Platform - Commands
echo.
echo Application Modes:
echo   run.bat web        - Start web interface (port 7860)
echo   run.bat csv        - Run CSV analysis mode
echo   run.bat process    - Run document processing mode
echo.
echo Service Management:
echo   run.bat stop       - Stop application (kill port 7860)
echo   run.bat kill-port  - Kill process on port 7860
echo   run.bat start-db   - Start PostgreSQL database (Docker)
echo   run.bat stop-db    - Stop PostgreSQL database (Docker)
echo.
echo Maintenance:
echo   run.bat clean      - Clean up Python cache files
goto end

:web
echo Starting web interface...
call :kill_port_process
call venv\Scripts\activate
python app/main.py --mode web
goto end

:csv
echo Starting CSV analysis mode...
call venv\Scripts\activate
python app/main.py --mode csv
goto end

:process
echo Starting document processing mode...
call venv\Scripts\activate
python app/main.py --mode process
goto end

:stop
echo Stopping application...
call :kill_port_process
echo Application stopped
goto end

:kill-port
echo Killing process on port 7860...
call :kill_port_process
goto end

:start-db
echo Starting PostgreSQL database...
docker-compose up -d postgres
goto end

:stop-db
echo Stopping PostgreSQL database...
docker-compose stop postgres
goto end

:clean
echo Cleaning Python cache files...
del /s /q *.pyc 2>nul
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo Cleanup complete
goto end

:kill_port_process
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860') do (
    echo Killing process on port 7860 (PID: %%a)
    taskkill /F /PID %%a
)
goto :eof

:end
