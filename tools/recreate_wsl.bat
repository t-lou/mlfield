@echo off
REM Script to remove WSL Ubuntu 24.04, reinstall it, and run install_docker.sh

echo.
echo ========================================
echo WSL Ubuntu 24.04 Recreation Script
echo ========================================
echo.

REM Ask for confirmation before removing
echo WARNING: This will remove your existing WSL Ubuntu 24.04 installation.
set /p confirm="Do you want to continue? (yes/no): "

if /i not "%confirm%"=="yes" (
    echo Cancelled.
    exit /b 0
)

echo.
echo Removing WSL Ubuntu 24.04...
wsl --unregister Ubuntu-24.04
if %errorlevel% neq 0 (
    echo Warning: Could not unregister Ubuntu-24.04. It may not exist.
)

echo.
echo Installing WSL Ubuntu 24.04...
wsl --install Ubuntu-24.04 --no-launch
if %errorlevel% neq 0 (
    echo Error: Failed to install Ubuntu-24.04
    exit /b 1
)

echo.
echo Running install_docker.sh in WSL...
wsl -d Ubuntu-24.04 bash -c "cd $(wslpath '%~dp0../.devcontainer') && bash install_docker.sh"
if %errorlevel% neq 0 (
    echo Error: Failed to run install_docker.sh
    exit /b 1
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
