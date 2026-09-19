# Sample C: free space, pagefile allocation and commit charge at a fine interval.
#
# The 60-second series (`reports/disk_watch-*.csv`) showed that C: free space moves
# almost perfectly inversely with the pagefile allocation (corr = -0.913) and that on
# a typical minute the pagefile explains the change completely (median residual
# -0.01 GB). But a few minutes carried residuals of 5-11 GB, and a 60-second cadence
# cannot tell a real second consumer of C: apart from the pagefile changing between
# two samples. This sampler runs at a few seconds per sample to separate the two.
#
# Two independent free-space readings are recorded on purpose: `Get-PSDrive` and the
# LogicalDisk perf counter. If they disagree, one of them is not live and the
# residual analysis below is measuring the tool rather than the machine.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sample_disk_fast.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sample_disk_fast.ps1 -Seconds 2 -Samples 300
param(
    [int]$Seconds = 2,
    [int]$Samples = 300,
    [string]$Root = 'C:\Users\heyiy\Desktop\cangying'
)

$ErrorActionPreference = 'Continue'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path $Root "reports\disk_fast-$stamp.csv"
"t,C_free_GB,C_free_perf_GB,pf_alloc_GB,pf_used_GB,commit_used_GB" |
    Out-File -FilePath $out -Encoding utf8

for ($i = 0; $i -lt $Samples; $i++) {
    $psdrive = (Get-PSDrive C).Free / 1GB
    $perf = (Get-Counter '\LogicalDisk(C:)\Free Megabytes').CounterSamples[0].CookedValue / 1024
    $pf = Get-CimInstance Win32_PageFileUsage | Measure-Object -Property AllocatedBaseSize -Sum
    $pfUsed = Get-CimInstance Win32_PageFileUsage | Measure-Object -Property CurrentUsage -Sum
    $os = Get-CimInstance Win32_OperatingSystem
    $commit = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB
    "{0},{1:N3},{2:N3},{3:N3},{4:N3},{5:N3}" -f (Get-Date -Format 'HH:mm:ss.fff'),
        $psdrive, $perf, ($pf.Sum / 1024), ($pfUsed.Sum / 1024), $commit |
        Out-File -FilePath $out -Append -Encoding utf8
    Start-Sleep -Seconds $Seconds
}

Write-Output "written: $out ($Samples samples at ${Seconds}s)"
