$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Config = Join-Path $PSScriptRoot "..\configs\rabat_real_pilot.json"

& $Python -m geoai_rabat.cli run-complete --config $Config
& $Python -m geoai_rabat.cli verify-delivery --config $Config
& $Python -m pytest
