@echo off
REM Build o_voxel CUDA extension with Visual Studio 2022
REM This script sets up the VS environment and builds the extension

set DISTUTILS_USE_SDK=1

REM Try to find VS2022
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo Found VS2022 Community
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    echo Found VS2022 Professional
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    echo Found VS2022 Enterprise
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo Found VS2019 Community
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    echo ERROR: Could not find Visual Studio. Please install Visual Studio 2022 or 2019 with C++ workload.
    exit /b 1
)

REM Verify cl.exe is available
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found after vcvarsall.bat. Please check your VS installation.
    exit /b 1
)

echo.
echo MSVC compiler found:
cl 2>&1 | findstr /C:"Microsoft"

cd /d "B:\M\ArtificialArchitecture\spatial\trellis-forge\o_voxel"

REM Clean previous build
if exist "build" rmdir /s /q "build"
if exist "o_voxel.egg-info" rmdir /s /q "o_voxel.egg-info"

echo.
echo Building o_voxel CUDA extension...
echo.

"B:\M\ArtificialArchitecture\spatial\trellis-forge\venv311\Scripts\python.exe" setup.py build_ext --inplace 2>&1

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

echo.
echo Installing o_voxel...
"B:\M\ArtificialArchitecture\spatial\trellis-forge\venv311\Scripts\pip.exe" install -e . --no-build-isolation

if errorlevel 1 (
    echo.
    echo INSTALL FAILED
    exit /b 1
)

echo.
echo BUILD AND INSTALL SUCCESSFUL
exit /b 0
