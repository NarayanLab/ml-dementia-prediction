@echo off
echo ===========================================
echo  Dementia Risk Assessment Tool
echo  Starting Backend and Frontend Servers
echo ===========================================
echo.

echo Starting Backend (FastAPI)...
start "Backend Server" cmd /k "cd backend && python main.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting Frontend (React)...
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo ===========================================
echo Both servers are starting!
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo The React app should open automatically
echo ===========================================
pause