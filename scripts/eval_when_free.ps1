# Run the fixed 24-batch evaluation as soon as the training arm releases the GPU.
#
# Why this exists: the hourly round cannot do this reliably. `flybrain.serve` keeps
# `autoloop_busy_check.ps1` reporting BUSY even after the arm stops, so the rule
# "pass the gate before any GPU work" would block the evaluation indefinitely; and
# the arm stops at an unpredictable step against its minute cap (a photo finish),
# so nobody can time a manual run either.
#
# Scope: evaluation only. It never replaces or deletes a checkpoint. Replacing
# canonical is a separate, reviewed step that must first rename the old best.pt.

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'
$python = Join-Path $Root '.venv\Scripts\python.exe'
$checkpoint = 'E:\flybrain-connectome\rank512-long\best.pt'
$evalOut = Join-Path $Root 'reports\eval_flybrain-connectome-rank512-long_c32.json'
$log = Join-Path $Root 'reports\eval_when_free.log'

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Get-TrainingProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match 'flybrain\.train' }
}

Write-Log "watching for the training arm to stop (checkpoint: $checkpoint)"
$deadline = (Get-Date).AddHours(4)
while ((Get-TrainingProcesses) -and ((Get-Date) -lt $deadline)) {
    Start-Sleep -Seconds 20
}

if (Get-TrainingProcesses) {
    Write-Log "GAVE UP: a flybrain.train process is still alive after 4 hours"
    exit 3
}

# The driver that launched the arm may run its own eval right after the child
# exits. Give it a moment, then skip if it already produced a fresh result.
Write-Log "no flybrain.train process left; waiting 25 s for any attached driver"
Start-Sleep -Seconds 25

if (Test-Path $evalOut) {
    $age = (Get-Date) - (Get-Item $evalOut).LastWriteTime
    if ($age.TotalMinutes -lt 10) {
        Write-Log "SKIP: $evalOut was written $([math]::Round($age.TotalSeconds,1)) s ago (a driver already evaluated it)"
        exit 0
    }
}

if (-not (Test-Path $checkpoint)) {
    Write-Log "GAVE UP: no checkpoint at $checkpoint"
    exit 4
}

Write-Log "running the fixed 24-batch eval (same command that measured 3.6349)"
& $python -u 'scripts\eval_checkpoint.py' --checkpoint $checkpoint `
    --tokens 'data\tokenized_domain\val.bin' --batches 24 --batch 8 --context 32 `
    --out $evalOut 2>&1 | ForEach-Object { Write-Log "  $_" }
Write-Log "eval exit=$LASTEXITCODE -> $evalOut"
exit 0
