$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Environnement absent. Exécutez d'abord scripts\setup.ps1."
}

Push-Location $ProjectRoot
try {
    & $Python -m geoai_rabat.cli run-week5 --config configs/rabat.json
    & $Python -m pytest
}
finally {
    Pop-Location
}
