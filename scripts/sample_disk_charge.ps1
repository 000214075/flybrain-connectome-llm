# Fine-grained (1-2 s) disk accounting probe: C: free space from two independent
# readings, the pagefile allocation, the commit charge, and the process-level
# private bytes that (together with RAM) make up that commit charge.
#
# Why two free-space readings: report 7.31/7.33 measured C: free space against the
# pagefile allocation, and a residual would mean either a second consumer of C: or a
# reading that is not live. `Get-PSDrive` already matched .NET `DriveInfo` to 0.000 GB
# over 45 samples, while the perf counter `\LogicalDisk(C:)\Free Megabytes` updated
# only ~3 times in 300 samples -- it is the stale one, so keep it out of the residual.
#
# `sum_private_gb` is the sum of PrivateMemorySize64 over every process: the C: space a
# training arm costs is a pagefile *reservation*, so the interesting quantity is commit
# (whose growth forces the pagefile to grow), not bytes written.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sample_disk_charge.ps1
param(
    [int]$Samples = 45,
    [double]$Seconds = 1.5,
    [string]$Root = 'C:\Users\heyiy\Desktop\cangying'
)

$ErrorActionPreference = 'Continue'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path $Root "reports\disk_charge-$stamp.csv"
"t,psdrive_free_gb,driveinfo_free_gb,pf_alloc_gb,pf_used_gb,commit_gb,sum_private_gb,python_private_gb,top_proc,top_private_gb" |
    Out-File -FilePath $out -Encoding utf8

for ($i = 0; $i -lt $Samples; $i++) {
    $ps = (Get-PSDrive C).Free / 1GB
    $di = (New-Object System.IO.DriveInfo('C')).AvailableFreeSpace / 1GB
    $pf = Get-CimInstance Win32_PageFileUsage | Measure-Object -Property AllocatedBaseSize -Sum
    $pfu = Get-CimInstance Win32_PageFileUsage | Measure-Object -Property CurrentUsage -Sum
    $os = Get-CimInstance Win32_OperatingSystem
    $commit = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB
    $procs = Get-Process
    $sumPriv = ($procs | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1GB
    $pyPriv = ($procs | Where-Object { $_.ProcessName -eq 'python' } | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1GB
    $top = $procs | Sort-Object -Property PrivateMemorySize64 -Descending | Select-Object -First 1
    "{0},{1:N3},{2:N3},{3:N3},{4:N3},{5:N3},{6:N3},{7:N3},{8},{9:N3}" -f (Get-Date -Format 'HH:mm:ss.fff'),
        $ps, $di, ($pf.Sum / 1024), ($pfu.Sum / 1024), $commit, $sumPriv, $pyPriv, $top.ProcessName, ($top.PrivateMemorySize64 / 1GB) |
        Out-File -FilePath $out -Append -Encoding utf8
    Start-Sleep -Milliseconds ([int]($Seconds * 1000))
}

Write-Output "written: $out ($Samples samples at ${Seconds}s)"
