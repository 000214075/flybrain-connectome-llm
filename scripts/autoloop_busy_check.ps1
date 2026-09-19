<#
Is another autoloop round already working in this project?

The loop has two engines (see autoloop/README.md): a global Stop hook and a scheduler
task. The hook refuses to inject when work is already in flight --

    # Something is already running: its completion will wake the session by itself,
    # so injecting here would only stack a second round on top of work in flight.

-- and there is a test for it (`test_work_already_in_flight_is_not_stacked_on`).
The scheduler has no such guard: it fires on wall clock and knows nothing about what
is running, so a round that takes longer than the interval gets a second round started
on top of it. That is not hypothetical here: rounds in this project routinely run
1-2 hours (training is capped at 40 minutes at a time) while the scheduler fires every
hour, so overlap is the normal case, not the exception.

Why it matters on this machine specifically: there is one GPU, and a second job on it
does not just slow the first one down, it invalidates its numbers. Measured on this
host, a training run's throughput fell from 356 tok/s to 51 tok/s while another process
held the card, and a wall-clock-capped run therefore completes a fraction of its steps.
That is precisely how this project produced a false "the shuffled control is 3.3x
slower" conclusion (reports/FINAL_REPORT.md section 7.6), which then had to be retracted.

Verdict on stdout, exit code carries the answer:
    0 = nothing in flight, start the round
    3 = work in flight, let this firing go

Fail-safe direction is FREE. Any error, missing directory or unreadable process list
reports free (exit 0): a check that wrongly says "busy" would stall the loop forever,
which is worse than an occasional overlap.

    powershell -ExecutionPolicy Bypass -File scripts\autoloop_busy_check.ps1
    powershell -ExecutionPolicy Bypass -File scripts\autoloop_busy_check.ps1 -QuietMinutes 10
#>
[CmdletBinding()]
param(
    [int]$QuietMinutes = 5,
    # Second way to recognise this project's jobs, for the case where the interpreter
    # path does not name the project: a venv launcher re-execs the *base* interpreter,
    # so `-m flybrain.train` can appear with a command line holding no project path at
    # all. Tests pass something that matches nothing so their result does not depend on
    # whatever happens to be running on the host.
    [string]$AlsoMatch = 'flybrain',
    # Processes that live in the project but are not work: they never end, so counting
    # them would report BUSY for ever and starve the loop of every firing. The local
    # search MCP server is the one that matters -- it is started by the client at
    # session start and runs until the session ends, and its command line contains the
    # project path, so it looks exactly like a job to the checks below. It also holds
    # no GPU memory, which is what this gate is really protecting.
    [string]$Exclude = 'localsearch[\\/]server\.py'
)

$ErrorActionPreference = 'SilentlyContinue'

$Project = if ($env:AUTOLOOP_PROJECT) { $env:AUTOLOOP_PROJECT } else { 'C:\Users\heyiy\Desktop\cangying' }

function Say-Free([string]$Why) {
    Write-Output "FREE: $Why"
    exit 0
}

function Say-Busy([string]$Why) {
    Write-Output "BUSY: $Why"
    exit 3
}

if (-not (Test-Path -LiteralPath $Project)) { Say-Free "project $Project does not exist" }

# --- signal 1: a python job belonging to this project is running ----------------
# Every project process counts, including `flybrain.serve`: a server holds GPU memory
# and a round that verifies the web interface is a round of work like any other, so
# stacking on top of it is exactly the mistake this check exists to prevent. The cost
# of being strict here is bounded -- a stale server only makes the *backup* engine skip
# a firing, while the Stop hook (the primary engine) keeps the loop alive and every
# round is already told to clear strays before any timing run.
$jobs = @()
try {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop
    foreach ($p in $procs) {
        $cmd = "$($p.CommandLine)"
        if ([string]::IsNullOrWhiteSpace($cmd)) { continue }
        if ($Exclude -and $cmd -match $Exclude) { continue }
        $mine = $cmd -like "*$Project*"
        if (-not $mine -and $AlsoMatch) { $mine = $cmd -match [regex]::Escape($AlsoMatch) }
        if ($mine) {
            $jobs += "pid $($p.ProcessId): $cmd"
        }
    }
} catch {
    Say-Free "could not read the process list ($($_.Exception.Message))"
}
if ($jobs.Count -gt 0) {
    Say-Busy "$($jobs.Count) project job(s) running -- $($jobs[0])"
}

# --- signal 2: the project was written to seconds ago --------------------------
# Every round writes reports/, configs/ or scripts/ within its first minute, so a very
# recent mtime means a round is in progress even if its compute is inside one process
# that the check above cannot name. This signal decays on its own, so it cannot wedge.
if ($QuietMinutes -le 0) { Say-Free "no project job running; recency check disabled" }
$cutoff = (Get-Date).AddMinutes(-$QuietMinutes)
$recent = @()
foreach ($sub in 'reports', 'configs', 'scripts', 'checkpoints') {
    $dir = Join-Path $Project $sub
    if (-not (Test-Path -LiteralPath $dir)) { continue }
    $hits = Get-ChildItem -LiteralPath $dir -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt $cutoff }
    if ($hits) { $recent += "$sub ($($hits.Count) file(s), newest $($hits | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.Name }) at $($hits | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.LastWriteTime.ToString('HH:mm:ss') }))" }
}
if ($recent.Count -gt 0) {
    Say-Busy "project files written in the last $QuietMinutes minute(s) -- $($recent -join '; ')"
}

Say-Free "no project job running and no file written in the last $QuietMinutes minute(s)"
