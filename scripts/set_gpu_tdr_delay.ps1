<#
Raise the Windows TDR timeout so a long GPU kernel cannot reset the display driver.

The problem
-----------
Windows watches the display driver. If a GPU operation blocks longer than `TdrDelay`
seconds -- effectively the preemption timeout -- Windows resets the driver. Microsoft
documents the default as **2 seconds**, and on this host the key is not set at all, so
the default applies. Intel's own oneAPI installation guide for Windows says the same
thing and tells GPU-compute users to raise it:

    "Windows resets GPUs if the compute kernels run longer than a few seconds. To
     avoid a device reset during long running kernels, [adjust TDR]."

For training this is the difference between "the run finished" and "the process died
with no Python traceback and the run had to be resumed". The connectome sweep is a
single sparse matmul over 25.5M edges per step, which is exactly the shape of work
that can sit on the GPU long enough to be noticed.

What this script does
---------------------
Sets, under HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers:

    TdrDelay    = 60     (seconds the GPU may block before recovery; default 2)
    TdrDdiDelay = 60     (seconds a thread may stay in the driver; default 5)

Raising the delay does not break recovery: a genuinely hung GPU is still reset, just
after 60 s instead of 2. `TdrLimitCount`/`TdrLimitTime` (default 5 crashes per 60 s)
are left alone, so repeated resets still bugcheck the machine the way they should.

Honest caveat
-------------
Microsoft's page says these keys are for driver developers and that end users should
not manipulate them. That is a statement about the supported path, not about the
effect: the value is read by the OS and the semantics are documented. This is a
documented, vendor-recommended workaround for long-running GPU compute on Windows,
and it is reversible with -Revert.

Must run from an **elevated** PowerShell: it writes to HKLM. This project's sessions
run unelevated, so the agent cannot apply it -- it is here for the user to run.

Usage:
    # from an Administrator PowerShell
    powershell -ExecutionPolicy Bypass -File scripts\set_gpu_tdr_delay.ps1
    powershell -ExecutionPolicy Bypass -File scripts\set_gpu_tdr_delay.ps1 -Revert
    powershell -ExecutionPolicy Bypass -File scripts\set_gpu_tdr_delay.ps1 -Status
#>
[CmdletBinding()]
param(
    [int]$Seconds = 60,
    [switch]$Revert,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'
$key = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'

$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

function Show-Status {
    $props = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
    foreach ($name in 'TdrLevel', 'TdrDelay', 'TdrDdiDelay', 'TdrLimitCount', 'TdrLimitTime') {
        $value = if ($null -ne $props -and $null -ne $props.$name) { $props.$name } else { '(unset)' }
        Write-Host ("  {0,-14} {1}" -f $name, $value)
    }
    $delay = if ($null -ne $props -and $null -ne $props.TdrDelay) { $props.TdrDelay } else { 2 }
    Write-Host ""
    Write-Host ("  effective TdrDelay: {0} s  (default when unset is 2)" -f $delay)
    if ($delay -le 2) {
        Write-Host "  -> long GPU kernels can still be reset as if hung."
    } else {
        Write-Host "  -> long GPU kernels get the raised budget."
    }
}

if ($Status) {
    Write-Host "TDR settings under $key"
    Show-Status
    exit 0
}

if (-not $elevated) {
    Write-Host "Not elevated: this script writes to HKEY_LOCAL_MACHINE and cannot continue." -ForegroundColor Yellow
    Write-Host "Re-run it from an Administrator PowerShell:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\set_gpu_tdr_delay.ps1"
    Write-Host ""
    Write-Host "Current values:"
    Show-Status
    exit 1
}

if ($Revert) {
    foreach ($name in 'TdrDelay', 'TdrDdiDelay') {
        if ($null -ne (Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).$name) {
            Remove-ItemProperty -Path $key -Name $name -ErrorAction SilentlyContinue
            Write-Host "removed $name"
        }
    }
    Write-Host "reverted to the Windows defaults (TdrDelay 2 s, TdrDdiDelay 5 s)"
    exit 0
}

if ($Seconds -lt 5 -or $Seconds -gt 900) {
    throw "-Seconds must be between 5 and 900 (got $Seconds)"
}

New-ItemProperty -Path $key -Name 'TdrDelay' -Value $Seconds -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $key -Name 'TdrDdiDelay' -Value $Seconds -PropertyType DWord -Force | Out-Null
Write-Host "set TdrDelay=$Seconds and TdrDdiDelay=$Seconds"
Write-Host ""
Write-Host "A reboot is NOT required for new processes; the value is read when the GPU"
Write-Host "scheduler is initialised. If a driver reset happens during this session anyway,"
Write-Host "re-run with -Status and check the Windows event log:"
Write-Host "  Get-WinEvent -FilterHashtable @{LogName='System'; Id=4101} -MaxEvents 10"
Write-Host ""
Show-Status
