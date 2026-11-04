# Build CuMesh for Windows with CUDA
$ErrorActionPreference = "Continue"

# Set paths
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$cumeshDir = $PSScriptRoot

# Create a temporary batch script
$batchContent = @"
@echo off
call "$vsPath" x64
if errorlevel 1 (
    echo Failed to initialize VS environment
    exit /b 1
)
echo VS Environment initialized

set CUDA_HOME=$cudaPath
set CUDA_PATH=$cudaPath
set PATH=$cudaPath\bin;%PATH%
set DISTUTILS_USE_SDK=1
echo CUDA_HOME=%CUDA_HOME%
echo DISTUTILS_USE_SDK=%DISTUTILS_USE_SDK%

cd /d $cumeshDir
echo Building cumesh...
pip install . --no-build-isolation -v
echo Build complete with exit code %errorlevel%
"@

$batchFile = Join-Path $cumeshDir "run_build.bat"
$batchContent | Out-File -FilePath $batchFile -Encoding ASCII

# Run the batch file
Write-Host "Starting build..."
& cmd.exe /c $batchFile 2>&1
Write-Host "Build finished with exit code: $LASTEXITCODE"
