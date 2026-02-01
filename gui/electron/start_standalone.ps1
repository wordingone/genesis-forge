# TRELLIS Standalone GUI Launcher
# Installs dependencies and starts the Electron application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TRELLIS Standalone GUI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$electronPath = $PSScriptRoot

# Check if node_modules exists
if (-not (Test-Path (Join-Path $electronPath "node_modules"))) {
    Write-Host "Installing Electron dependencies..." -ForegroundColor Yellow
    Write-Host ""

    Push-Location $electronPath
    npm install
    Pop-Location

    Write-Host ""
    Write-Host "Dependencies installed!" -ForegroundColor Green
    Write-Host ""
}

# Start Electron app
Write-Host "Starting TRELLIS GUI..." -ForegroundColor Green
Write-Host ""

Push-Location $electronPath
npm start
Pop-Location
