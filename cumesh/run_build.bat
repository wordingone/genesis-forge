@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo Failed to initialize VS environment
    exit /b 1
)
echo VS Environment initialized

set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%
set DISTUTILS_USE_SDK=1
echo CUDA_HOME=%CUDA_HOME%
echo DISTUTILS_USE_SDK=%DISTUTILS_USE_SDK%

cd /d B:\M\ArtificialArchitecture\spatial\trellis-forge\cumesh
echo Building cumesh...
pip install . --no-build-isolation -v
echo Build complete with exit code %errorlevel%
