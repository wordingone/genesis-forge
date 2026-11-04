# Build o_voxel CUDA extension
$ErrorActionPreference = "Continue"
$env:DISTUTILS_USE_SDK = "1"

Set-Location $PSScriptRoot

# Clean previous build
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "o_voxel.egg-info") { Remove-Item -Recurse -Force "o_voxel.egg-info" }

$pythonExe = (Join-Path $PSScriptRoot "..\venv\Scripts\python.exe")
$pipExe = (Join-Path $PSScriptRoot "..\venv\Scripts\pip.exe")

# Build
Write-Host "Building o_voxel..."
& $pythonExe setup.py build_ext --inplace 2>&1 | Tee-Object -FilePath "build.log"
$buildResult = $LASTEXITCODE

Write-Host ""
Write-Host "Build completed with exit code: $buildResult"

if ($buildResult -eq 0) {
    Write-Host "Installing..."
    & $pipExe install -e . --no-build-isolation 2>&1
}

exit $buildResult
