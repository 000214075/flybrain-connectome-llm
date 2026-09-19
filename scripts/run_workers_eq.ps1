# Experiment D: does `num_workers: 0` leave the data order untouched?
#
# Each training arm currently gives ~19.5 GB of commit charge to four dataloader
# workers that exist only to slice an 80 MB token array -- each worker is a fresh
# Python process that re-imports torch/ROCm (FINAL_REPORT.md section 7.24). Dropping
# them would relieve a machine that has been measured at 99.4% of its commit limit.
#
# The change is only free if the sample order does not move, so the arms are run and
# then compared step by step by `scripts/compare_metrics.py`: identical `loss` at every
# step passes, a single differing step fails. Throughput is reported separately,
# because saving memory is allowed to cost speed -- that is not the question here.
#
# Launch detached via WMI (see section 7.23 for why not `Start-Process`):
#
#   powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\heyiy\Desktop\cangying\scripts\run_workers_eq.ps1'; CurrentDirectory = 'C:\Users\heyiy\Desktop\cangying' }"
param(
    [string]$Root = 'C:\Users\heyiy\Desktop\cangying',
    # Config base names to train (without .json), in order.
    [string[]]$Run = @('workers_eq2', 'workers_eq0'),
    # The two metrics.jsonl files to compare once training is done.
    [string]$CompareA = 'E:\flybrain-workers\eq2\metrics.jsonl',
    [string]$CompareB = 'E:\flybrain-workers\eq0\metrics.jsonl',
    [string]$ResultFile = 'reports\workers_eq_compare.txt'
)

$ErrorActionPreference = 'Continue'
Set-Location $Root
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'

$python = Join-Path $Root '.venv\Scripts\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $Root "reports\workers_eq-$stamp.log"
$lock = Join-Path $Root 'reports\workers_eq.lock'
$resultFile = Join-Path $Root $ResultFile

function Write-Log([string]$text) {
    Add-Content -Path $log -Value ((Get-Date -Format 'HH:mm:ss') + ' ' + $text) -Encoding ASCII
}

# One GPU: refuse to start while another sweep or run of this experiment is alive.
foreach ($other in 'rank_sweep.lock', 'workers_eq.lock') {
    $path = Join-Path $Root "reports\$other"
    if (Test-Path $path) {
        $holder = Get-Content $path -TotalCount 1
        if ($holder -and (Get-Process -Id $holder -ErrorAction SilentlyContinue)) {
            Write-Output "BUSY: $other is held by live PID $holder"
            exit 3
        }
    }
}
Set-Content -Path $lock -Value $PID -Encoding ASCII

try {
    Write-Log "workers-equivalence start pid=$PID log=$log"

    foreach ($name in $Run) {
        $config = Join-Path $Root "configs\$name.json"
        if (-not (Test-Path $config)) {
            Write-Log "=== $name MISSING CONFIG $config (skipping) ==="
            continue
        }
        # PowerShell 5.1 decodes without -Encoding UTF8 as the system ANSI codepage,
        # which breaks ConvertFrom-Json on the config's Chinese sample_prompt and
        # returns nothing (section 7.22). Read the path out of the config instead of
        # rebuilding it, so there is one source of truth (section 7.21).
        $outDir = (Get-Content $config -Raw -Encoding UTF8 | ConvertFrom-Json).out_dir
        if (-not $outDir) {
            Write-Log "=== $name COULD NOT READ out_dir (skipping) ==="
            continue
        }

        Write-Log "=== $name train start (out_dir=$outDir) ==="
        & $python -u -m flybrain.train --config $config --resume none 2>&1 |
            Out-File -Append -Encoding utf8 -FilePath $log
        Write-Log "=== $name train exit=$LASTEXITCODE (exit code is not a verdict) ==="
    }

    if ((Test-Path $CompareA) -and (Test-Path $CompareB)) {
        $comparison = & $python scripts\compare_metrics.py --a $CompareA --b $CompareB 2>&1
        $code = $LASTEXITCODE
        $comparison | Out-File -FilePath $resultFile -Encoding utf8
        $comparison | ForEach-Object { Write-Log $_ }
        Write-Log "=== compare exit=$code (0 = identical step by step) -> $resultFile ==="
    } else {
        Write-Log "=== compare SKIPPED: missing metrics file ==="
        Write-Log "  a exists=$(Test-Path $CompareA) b exists=$(Test-Path $CompareB)"
    }
} finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}
