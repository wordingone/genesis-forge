# TRELLIS GUI Startup Script
# Starts both backend and frontend servers

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TRELLIS GUI Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Paths
$backendPath = Join-Path $PSScriptRoot "backend"
$frontendPath = Join-Path $PSScriptRoot "frontend"
$pythonExe = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"

# Verify paths exist
if (-not (Test-Path $backendPath)) {
    Write-Host "ERROR: Backend directory not found at $backendPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $frontendPath)) {
    Write-Host "ERROR: Frontend directory not found at $frontendPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $pythonExe)) {
    Write-Host "ERROR: Python not found at $pythonExe" -ForegroundColor Red
    Write-Host "Please verify your virtual environment is set up correctly" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting Backend Server..." -ForegroundColor Green
Write-Host "  Location: $backendPath" -ForegroundColor Gray
Write-Host "  Command: uvicorn main:app --host 127.0.0.1 --port 8000 --reload" -ForegroundColor Gray
Write-Host ""

# Start backend in new window
$backendCmd = "cd '$backendPath'; Write-Host 'TRELLIS Backend Server' -ForegroundColor Cyan; Write-Host ''; & '$pythonExe' -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

# Wait for backend to start
Write-Host "Waiting for backend to initialize (5 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if backend is responsive
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/docs" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    Write-Host "Backend is running on http://127.0.0.1:8000" -ForegroundColor Green
} catch {
    Write-Host "Backend may not be ready yet (this is normal on first start)" -ForegroundColor Yellow
    Write-Host "  Check the backend window for status" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Starting Frontend Server..." -ForegroundColor Green
Write-Host "  Location: $frontendPath" -ForegroundColor Gray
Write-Host "  Command: npm run dev" -ForegroundColor Gray
Write-Host ""

# Start frontend in new window
$frontendCmd = "cd '$frontendPath'; Write-Host 'TRELLIS Frontend Server' -ForegroundColor Cyan; Write-Host ''; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

# Wait a moment
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Servers Started!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend:  " -NoNewline -ForegroundColor White
Write-Host "http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "          API Docs: " -NoNewline -ForegroundColor White
Write-Host "http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Frontend: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
Write-Host "Opening browser..." -ForegroundColor Green
Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "To stop the servers:" -ForegroundColor White
Write-Host "  Press Ctrl+C in each server window" -ForegroundColor Gray
Write-Host ""
Write-Host "For help, see:" -ForegroundColor White
Write-Host "  gui/QUICKSTART.md" -ForegroundColor Gray
Write-Host "  gui/README.md" -ForegroundColor Gray
Write-Host ""
