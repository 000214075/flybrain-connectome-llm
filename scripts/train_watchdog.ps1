<#
Supervise a training run so that a crash costs a restart instead of the run.

Why this exists: on this machine the GPU is an RX 7900 XTX running ROCm on native
Windows, and two things end training runs early that are not the model's fault:

  * a TDR -- Windows resets a display driver whose operation blocks for longer than
    `TdrDelay` seconds, which defaults to 2 and is unset on this host. The training
    process then dies with a device-lost style error and no Python traceback;
  * the session that launched the run going away, which kills every child process.

Neither is a reason to lose the run: `resume: auto` already makes the trainer pick up
from `last.pt`. This script supplies the missing half -- decide *why* the child died,
and start it again unless it finished.

It also refuses to spin: if the same failure repeats faster than the checkpoint
interval, resuming cannot make progress, so it stops and says so instead of
relaunching forever.

All output is ASCII on purpose. A .ps1 written without a BOM is read as ANSI by
Windows PowerShell 5.1, which corrupts non-ASCII literals before the script ever
runs -- that bug has bitten this project twice.

Usage:
    powershell -File scripts\train_watchdog.ps1 -Config configs\train_connectome_scaled_long.json
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [int]$MaxRestarts = 50,
    [int]$CooldownSeconds = 20,
    [string]$LogDir = "reports",
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "no interpreter at $python" }
if (-not (Test-Path $Config)) { throw "no config at $Config" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$watchLog = Join-Path $LogDir "watchdog-$stamp.log"

# Single instance, by PID lockfile. This script is launched by a repeating scheduled
# task and its own loop is the supervision, so a second copy would put a second job on
# the one GPU -- which does not merely slow the first run down, it invalidates its
# throughput numbers.
#
# The first version of this guard scanned the process list for powershell command lines
# matching the script name. That was wrong in a way worth recording: any shell whose
# command line merely *mentions* the path matched, including the launcher that starts
# the watchdog. The guard then reported "another watchdog is already running" and the
# real watchdog exited immediately, wrote no log, and started nothing -- silently, and
# with exit code 0. A PID in a file cannot be confused by command-line text.
$lockFile = Join-Path $LogDir "watchdog.lock"
# Every guard decision goes to one append-only file. Without this, "the guard
# correctly declined" and "the guard never ran" leave the same evidence -- nothing --
# because a blocked instance exits before it creates its per-instance log. That made
# a working scheduled task indistinguishable from a dead one during debugging.
$guardLog = Join-Path $LogDir "watchdog-guard.log"
function Write-Guard([string]$message) {
    Add-Content -LiteralPath $guardLog -Encoding UTF8 `
        -Value ("{0} pid={1} {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $PID, $message)
}

if (Test-Path -LiteralPath $lockFile) {
    $previous = (Get-Content -LiteralPath $lockFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    $previousPid = 0
    if ([int]::TryParse("$previous", [ref]$previousPid) -and $previousPid -ne $PID) {
        if (Get-Process -Id $previousPid -ErrorAction SilentlyContinue) {
            Write-Guard "blocked: pid $previousPid is alive and holds the lock"
            Write-Host "another watchdog is already running (pid $previousPid); exiting"
            exit 0
        }
        Write-Guard "stale lock from pid $previousPid (not running); taking over"
        Write-Host "stale lock from pid $previousPid (not running); taking over"
    }
}
Set-Content -LiteralPath $lockFile -Value $PID -Encoding ASCII
Write-Guard "acquired lock; launching training"

function Remove-Lock {
    # Only the owner removes it: a later instance that took over a stale lock must not
    # have its lock deleted by the older process finishing up.
    $current = (Get-Content -LiteralPath $lockFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ("$current" -eq "$PID") { Remove-Item -LiteralPath $lockFile -ErrorAction SilentlyContinue }
}
trap { Remove-Lock; break }

function Write-Watch([string]$message) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Add-Content -Path $watchLog -Value $line -Encoding UTF8
    if (-not $Quiet) { Write-Host $line }
}

# Failure signatures worth naming, because "exit code 1" is ambiguous on this stack:
# torch routes its CSR beta warning to stderr, and PowerShell 5.1 surfaces that as a
# NativeCommandError with exit code 1 even when the run succeeded. So the exit code
# alone can never be the test -- the log has to be read.
$signatures = [ordered]@{
    'out of memory'      = 'HIP/CUDA out of memory'
    'device lost'        = 'device lost (TDR or driver reset)'
    'hipErrorNoDevice'   = 'no HIP device visible (driver reset)'
    'HSA_STATUS_ERROR'   = 'HSA runtime error (driver reset)'
    'TDR'                = 'TDR mentioned in output'
    'RuntimeError'       = 'PyTorch runtime error'
    'Traceback'          = 'Python traceback'
}

function Get-Verdict([string[]]$logPaths) {
    # Both streams, because the failure signatures and the success summary are not
    # guaranteed to be on the same one.
    $present = @($logPaths | Where-Object { Test-Path $_ })
    if ($present.Count -eq 0) { return @{ Done = $false; Reason = 'no log written' } }
    $text = ($present | ForEach-Object {
        Get-Content -Path $_ -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    }) -join "`n"
    if (-not $text) { return @{ Done = $false; Reason = 'empty log' } }

    # A completed run writes this summary; its presence is the positive signal that
    # the run reached its budget rather than dying.
    if ($text -match '"steps"\s*:\s*\d+' -and $text -match '"minutes"\s*:\s*[\d.]+') {
        return @{ Done = $true; Reason = 'run reported its summary' }
    }
    foreach ($key in $signatures.Keys) {
        if ($text -match [regex]::Escape($key)) {
            return @{ Done = $false; Reason = $signatures[$key] }
        }
    }
    return @{ Done = $false; Reason = 'exited without a summary and without a recognised signature' }
}

Write-Watch "watchdog start  config=$Config  max_restarts=$MaxRestarts"
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'

$restarts = 0
$fastFailures = 0
# try/finally rather than `trap`: every way out of this loop is an `exit`, and a
# finally block runs for `exit` while a trap does not. Without it a normal completion
# would leave the lock behind, and the next trigger would find a dead PID, log
# "stale lock" and take over -- correct, but noisy every single time.
try {
while ($true) {
    $runLog = Join-Path $LogDir "watchdog-run-$stamp-$restarts.log"
    $errLog = Join-Path $LogDir "watchdog-run-$stamp-$restarts.err.log"
    Write-Watch "launch attempt $restarts -> $runLog"
    $started = Get-Date

    # Start-Process with -RedirectStandardOutput writes the child's output straight to
    # the file. PowerShell's own `*>` capture buffers a native command's output and
    # would leave the log empty until the process exits -- which, for a run that is
    # meant to last hours, makes it impossible to tell "training" from "stuck".
    # Resume is the whole point: without it a restart would throw the run away.
    #
    # Deliberately NOT `-Wait`. Measured: with redirected output, `-Wait` stayed
    # blocked after the child had already died, so the supervisor sat inside the call
    # while the run was gone -- alive-looking, doing nothing, and never relaunching.
    # Polling the pid cannot block in that way.
    $proc = Start-Process -FilePath $python -PassThru -NoNewWindow `
        -RedirectStandardOutput $runLog -RedirectStandardError $errLog `
        -ArgumentList @('-u', '-m', 'flybrain.train', '--config', $Config, '--resume', 'auto')
    while (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 5
    }
    $code = -1
    try { $code = $proc.ExitCode } catch { $code = -1 }
    $elapsed = [int]((Get-Date) - $started).TotalSeconds

    $verdict = Get-Verdict @($runLog, $errLog)
    Write-Watch ("attempt $restarts ended: exit=$code elapsed=${elapsed}s done=$($verdict.Done) reason=$($verdict.Reason)")

    if ($verdict.Done) {
        Write-Watch "run completed; watchdog exiting"
        exit 0
    }

    # A crash that repeats before the trainer could have written a checkpoint means
    # resuming cannot help, so stop rather than loop.
    if ($elapsed -lt 120) {
        $fastFailures++
        Write-Watch "  fast failure $fastFailures of 3"
        if ($fastFailures -ge 3) {
            Write-Watch "three fast failures in a row; stopping instead of looping"
            exit 2
        }
    } else {
        $fastFailures = 0
    }

    $restarts++
    if ($restarts -gt $MaxRestarts) {
        Write-Watch "restart budget exhausted ($MaxRestarts); stopping"
        exit 3
    }

    # Give a reset driver time to come back before asking for the GPU again.
    Write-Watch "  waiting ${CooldownSeconds}s before resuming"
    Start-Sleep -Seconds $CooldownSeconds

    $probe = & $python -c "import torch; torch.cuda.init(); print('device ok', torch.cuda.get_device_name(0))" 2>&1
    Write-Watch "  gpu probe: $($probe -join ' ')"
}
} finally {
    Remove-Lock
}
