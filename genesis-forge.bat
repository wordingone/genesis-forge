@echo off
REM Genesis Forge Launcher - Run from anywhere
REM Usage: genesis-forge [options]
REM   No args    = Start full app (backend + electron)
REM   -backend   = Start backend only
REM   -electron  = Start electron only

powershell -ExecutionPolicy Bypass -File "%~dp0Start-GenesisForge.ps1" %*
