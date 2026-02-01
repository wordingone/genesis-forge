<#
.SYNOPSIS
    Genesis Forge Global Launcher
.DESCRIPTION
    Run from any PowerShell window to start Genesis Forge
.PARAMETER BackendOnly
    Start only the FastAPI backend server
.PARAMETER ElectronOnly
    Start only the Electron app (requires backend running)
.PARAMETER Gradio
    Start the Gradio web interface instead
.EXAMPLE
    Start-GenesisForge
    Start-GenesisForge -Gradio
    Start-GenesisForge -BackendOnly
#>

param(
    [switch]$BackendOnly,
    [switch]$ElectronOnly,
    [switch]$Gradio
)

$ErrorActionPreference = "Stop"

# === CONFIGURATION (relative to script location) ===
$GENESIS_ROOT = $PSScriptRoot
$VENV_PYTHON = Join-Path $GENESIS_ROOT "venv\Scripts\python.exe"
$BACKEND_DIR = Join-Path $GENESIS_ROOT "gui\backend"
$ELECTRON_DIR = Join-Path $GENESIS_ROOT "gui\electron"
$VCVARS64 = if ($env:VCVARS64) { $env:VCVARS64 } else { "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" }

# === BANNER ===
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Genesis Forge v1.0 - 3D Generation                  " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# === VERIFY ENVIRONMENT ===
function Test-Prerequisites {
    $errors = @()

    if (-not (Test-Path $GENESIS_ROOT)) {
        $errors += "Genesis root not found: $GENESIS_ROOT"
    }

    if (-not (Test-Path $VENV_PYTHON)) {
        $errors += "Python venv not found: $VENV_PYTHON"
        $errors += "  Run: python -m venv venv && venv\Scripts\pip install -r requirements.txt"
    }

    if (-not (Test-Path $BACKEND_DIR)) {
        $errors += "Backend directory not found: $BACKEND_DIR"
    }

    if (-not (Test-Path $ELECTRON_DIR)) {
        $errors += "Electron directory not found: $ELECTRON_DIR"
    }

    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) {
        $errors += "npm not found in PATH"
    }

    if ($errors.Count -gt 0) {
        Write-Host "ERROR: Prerequisites not met:" -ForegroundColor Red
        $errors | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
        exit 1
    }

    Write-Host "[OK] Prerequisites verified" -ForegroundColor Green
}

# === GRADIO MODE ===
function Start-GradioApp {
    Write-Host ""
    Write-Host "Starting Gradio App (Image-to-3D)..." -ForegroundColor Yellow
    Write-Host "  URL: http://127.0.0.1:7860" -ForegroundColor Gray

    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$GENESIS_ROOT'; Write-Host 'Genesis Forge - Gradio' -ForegroundColor Cyan; & '$VENV_PYTHON' app.py"

    Start-Sleep -Seconds 3
    Start-Process "http://127.0.0.1:7860"
}

# === BACKEND SERVER ===
function Start-Backend {
    Write-Host ""
    Write-Host "Starting FastAPI Backend..." -ForegroundColor Yellow
    Write-Host "  API: http://127.0.0.1:8000" -ForegroundColor Gray
    Write-Host "  Docs: http://127.0.0.1:8000/docs" -ForegroundColor Gray

    # Check for MSVC (required for gsplat JIT compilation)
    if (Test-Path $VCVARS64) {
        Write-Host "  MSVC: Enabled (vcvars64.bat)" -ForegroundColor Green
        # Use cmd to call vcvars64.bat and then start uvicorn with MSVC environment
        $cmd = "call `"$VCVARS64`" && echo [MSVC] Environment configured && cd /d `"$BACKEND_DIR`" && `"$VENV_PYTHON`" -m uvicorn main:app --host 127.0.0.1 --port 8000"
        Start-Process cmd -ArgumentList "/k", "title Genesis Backend (MSVC) && $cmd"
    } else {
        Write-Host "  MSVC: Not found (textures will be black!)" -ForegroundColor Red
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$BACKEND_DIR'; Write-Host 'Genesis Backend (NO MSVC - textures broken)' -ForegroundColor Red; & '$VENV_PYTHON' -m uvicorn main:app --host 127.0.0.1 --port 8000"
    }
}

# === ELECTRON APP ===
function Start-Electron {
    Write-Host ""
    Write-Host "Starting Electron App..." -ForegroundColor Yellow

    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ELECTRON_DIR'; Write-Host 'Genesis Electron' -ForegroundColor Cyan; npm start"
}

# === WAIT FOR BACKEND ===
function Wait-ForBackend {
    Write-Host "Waiting for backend..." -ForegroundColor Gray -NoNewline
    $maxAttempts = 20

    for ($i = 0; $i -lt $maxAttempts; $i++) {
        try {
            $null = Invoke-WebRequest -Uri "http://127.0.0.1:8000/docs" -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
            Write-Host " Ready!" -ForegroundColor Green
            return $true
        } catch {
            Write-Host "." -NoNewline -ForegroundColor Gray
            Start-Sleep -Milliseconds 500
        }
    }

    Write-Host ""
    Write-Host "[WARN] Backend still loading (check window)" -ForegroundColor Yellow
    return $false
}

# === MAIN ===
Test-Prerequisites

if ($Gradio) {
    Start-GradioApp
    Write-Host ""
    Write-Host "Gradio starting. Press Ctrl+C in that window to stop." -ForegroundColor Green
}
elseif ($BackendOnly) {
    Start-Backend
    Write-Host ""
    Write-Host "Backend starting in new window." -ForegroundColor Green
}
elseif ($ElectronOnly) {
    Write-Host "[WARN] Electron requires backend!" -ForegroundColor Yellow
    Start-Electron
}
else {
    # Full stack
    Start-Backend
    Start-Sleep -Seconds 2
    Wait-ForBackend
    Start-Electron

    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Green
    Write-Host "  Genesis Forge Started                               " -ForegroundColor Green
    Write-Host "  Backend: http://127.0.0.1:8000                      " -ForegroundColor Green
    Write-Host "======================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To stop: Close Electron + Ctrl+C in backend window" -ForegroundColor Gray
}

Write-Host ""
