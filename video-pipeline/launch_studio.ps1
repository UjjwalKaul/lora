# PowerShell Launcher for Local Video Studio
# Guarantees no --reload flag to ensure safe Windows subprocess spawning

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "          Starting Local Video Studio Dashboard           " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Resolve Python executable
$PipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path (Split-Path -Parent $PipelineRoot) ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "[OK] Using virtualenv Python: $PythonExe" -ForegroundColor Green
} else {
    $PythonExe = "python"
    Write-Host "[WARN] Virtualenv not found at $VenvPython; falling back to system python" -ForegroundColor Yellow
}

# 2. Check Port 8000 availability
$PortOccupied = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($PortOccupied) {
    Write-Host "[WARN] Port 8000 appears to be in use. Please terminate existing servers or check PID: $($PortOccupied.OwningProcess)" -ForegroundColor Yellow
}

# 3. Launch browser in background after short delay
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:8000"
} | Out-Null

Write-Host "[INFO] Starting Uvicorn server on http://127.0.0.1:8000 (No reload)..." -ForegroundColor Green
Write-Host "[INFO] Press Ctrl+C to terminate the studio server." -ForegroundColor Gray

# Set working directory to video-pipeline so studio.server is cleanly imported
Set-Location $PipelineRoot
& $PythonExe -m uvicorn studio.server:app --host 127.0.0.1 --port 8000
