# Starts the FilingLens API and website in their own windows, then opens the browser.
# Usage (from the project folder):  .\dev.ps1
# Close the two windows (or press Ctrl+C in them) to stop.

$root = $PSScriptRoot

# uv may be a pip --user install that isn't on PATH yet.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $uvDir = Join-Path $env:APPDATA "Python\Python313\Scripts"
    if (Test-Path (Join-Path $uvDir "uv.exe")) { $env:PATH = "$uvDir;$env:PATH" }
    else { Write-Error "uv not found. Install it: powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""; exit 1 }
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Write-Error "Node.js/npm not found: install Node 22 from nodejs.org"; exit 1 }

function Test-Port([int]$port) {
    [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

# 1. API on :8000 (also starts the local database when DATABASE_URL="local").
if (Test-Port 8000) {
    Write-Host "API already running on http://127.0.0.1:8000"
} else {
    Write-Host "Starting the API..."
    Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle = 'FilingLens API'; `$env:PATH = '$env:PATH'; uv run filinglens api serve"
}

# 2. Website on :3000.
if (Test-Port 3000) {
    Write-Host "Website already running on http://localhost:3000"
} else {
    if (-not (Test-Path (Join-Path $root "web\node_modules"))) {
        Write-Host "Installing website dependencies (first run only)..."
        Push-Location (Join-Path $root "web"); npm ci; Pop-Location
    }
    Write-Host "Starting the website..."
    Start-Process powershell -WorkingDirectory (Join-Path $root "web") -ArgumentList "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle = 'FilingLens web'; npm run dev"
}

# 3. Wait until both answer, then open the browser.
Write-Host "Waiting for the servers" -NoNewline
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    try {
        Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/api/health" -TimeoutSec 2 | Out-Null
        Invoke-WebRequest -UseBasicParsing "http://localhost:3000/" -TimeoutSec 5 | Out-Null
        $ready = $true; break
    } catch { Write-Host "." -NoNewline; Start-Sleep -Seconds 1 }
}
Write-Host ""
if ($ready) {
    Write-Host "Opening http://localhost:3000"
    Start-Process "http://localhost:3000"
} else {
    Write-Warning "Not ready after 90 s. Check the 'FilingLens API' and 'FilingLens web' windows for errors."
}
