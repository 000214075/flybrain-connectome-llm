# Experiment B: the `brain_rank` sweep, one arm at a time.
#
# This machine has a single GPU, so the arms must not overlap; the sweep is
# sequential on purpose. Every arm is judged by the same fixed 24-batch eval
# command, never by the numbers the training process prints about itself, because
# the in-training eval uses `eval_iters` batches and carries a measured +0.19..+0.22
# offset against the fixed one.
#
# Launch it detached, NOT with -Wait, and NOT from a session shell. The first attempt
# used `Start-Process -WindowStyle Hidden`; the driver and its python child were both
# killed at 10:10:22 after 305 steps, with no traceback and without the `finally`
# below running (the lock file was left behind, so the process was terminated, not
# exited). The cause was not established -- no GPU driver reset, no session restart.
# So launch it through WMI instead, which parents the driver to WmiPrvSE.exe and
# therefore outside this session's process tree:
#
#   powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\heyiy\Desktop\cangying\scripts\run_rank_sweep.ps1'; CurrentDirectory = 'C:\Users\heyiy\Desktop\cangying' }"
#
# This is a precaution, not a diagnosis: earlier long runs launched with Start-Process
# did survive for hours (FINAL_REPORT.md section 7.20), so the pattern alone does not
# explain the kill.
#
# The training exit code is not a verdict. This stack returns 1 for runs that
# finished normally, because torch's CSR-beta warning goes to stderr. Read the
# per-arm eval JSON under reports\, not the exit code.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_rank_sweep.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_rank_sweep.ps1 -Ranks 128,512
param(
    [int[]]$Ranks = @(128, 256, 512),
    [string]$Root = 'C:\Users\heyiy\Desktop\cangying',
    # Explicit config paths, for runs that are not part of the rank sweep (e.g. the
    # 18000-step rank512 arm). When given, `Ranks` is ignored and each run's eval file
    # is named after the config's own run_name instead of a rank.
    [string[]]$Configs = @()
)

# `-File` passes `-Configs a,b,c` as one string, because comma binding is an
# in-session language feature. Split it here instead of looking for one impossible
# path — that failure is silent apart from the MISSING CONFIG line.
$Configs = @($Configs | ForEach-Object { $_ -split ',' } | Where-Object { $_ })

$ErrorActionPreference = 'Continue'
Set-Location $Root
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'

$python = Join-Path $Root '.venv\Scripts\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $Root "reports\rank_sweep-$stamp.log"
$lock = Join-Path $Root 'reports\rank_sweep.lock'

function Write-Log([string]$text) {
    Add-Content -Path $log -Value ((Get-Date -Format 'HH:mm:ss') + ' ' + $text) -Encoding ASCII
}

# Single instance: a second sweep would put two training jobs on one GPU.
if (Test-Path $lock) {
    $existing = Get-Content $lock -TotalCount 1
    if ($existing -and (Get-Process -Id $existing -ErrorAction SilentlyContinue)) {
        Write-Output "BUSY: rank sweep already running as PID $existing"
        exit 3
    }
}
Set-Content -Path $lock -Value $PID -Encoding ASCII

try {
    # Rank arms are named by rank; explicit configs are named by their own run_name,
    # so a long run and a sweep arm cannot write to the same eval file.
    $jobs = if ($Configs.Count -gt 0) {
        $Configs | ForEach-Object {
            $path = if ([IO.Path]::IsPathRooted($_)) { $_ } else { Join-Path $Root $_ }
            [pscustomobject]@{ Rank = $null; Config = $path }
        }
    } else {
        $Ranks | ForEach-Object {
            [pscustomobject]@{ Rank = $_; Config = (Join-Path $Root "configs\train_connectome_rank$_.json") }
        }
    }

    Write-Log "run start ranks=$($Ranks -join ',') configs=$($Configs -join ',') pid=$PID log=$log"
    foreach ($job in $jobs) {
        $rank = $job.Rank
        $config = $job.Config
        $label = if ($rank) { "rank $rank" } else { Split-Path $config -Leaf }
        if (-not (Test-Path $config)) {
            Write-Log "=== $label MISSING CONFIG $config (skipping) ==="
            continue
        }
        # Read out_dir from the config rather than rebuilding it here: when the driver
        # and the config each have their own idea of where checkpoints go, they can
        # disagree, and the mismatch is silent until an eval reads the wrong path.
        # Section 7.21 is that failure happening for real (a renamed directory while
        # the launcher still pointed at the old name).
        #
        # `-Encoding UTF8` is required, not cosmetic: in PowerShell 5.1 the default is
        # the system ANSI codepage, which mis-decodes the config's Chinese sample_prompt
        # and makes ConvertFrom-Json fail with a bogus syntax error. Without it this
        # read returns nothing and the eval silently finds no checkpoint.
        $cfg = Get-Content $config -Raw -Encoding UTF8 | ConvertFrom-Json
        $outDir = $cfg.out_dir
        if (-not $outDir) {
            Write-Log "=== $label COULD NOT READ out_dir FROM $config (skipping) ==="
            continue
        }
        if (-not [IO.Path]::IsPathRooted($outDir)) { $outDir = Join-Path $Root $outDir }
        $tag = if ($rank) { "rank$rank" } else { $cfg.run_name }

        Write-Log "=== $label train start (out_dir=$outDir) ==="
        # `2>&1 | Out-File -Encoding utf8`, not `*>>`: in PowerShell 5.1 `*>>` writes
        # UTF-16, which interleaves NUL bytes with the ASCII lines Write-Log adds.
        & $python -u -m flybrain.train --config $config --resume auto 2>&1 |
            Out-File -Append -Encoding utf8 -FilePath $log
        Write-Log "=== $label train exit=$LASTEXITCODE (see note: exit code is not a verdict) ==="

        $checkpoint = Join-Path $outDir 'best.pt'
        if (Test-Path $checkpoint) {
            $evalOut = "reports\eval_${tag}_c32.json"
            & $python -u scripts\eval_checkpoint.py --checkpoint $checkpoint `
                --tokens data\tokenized_domain\val.bin --batches 24 --batch 8 `
                --context 32 --out $evalOut 2>&1 |
                Out-File -Append -Encoding utf8 -FilePath $log
            Write-Log "=== $label eval exit=$LASTEXITCODE -> $evalOut ==="
        } else {
            Write-Log "=== $label NO CHECKPOINT at $checkpoint, eval skipped ==="
        }
    }

    Write-Log "run finished; fixed 24-batch results:"
    foreach ($job in $jobs) {
        $tag = if ($job.Rank) { "rank$($job.Rank)" } else { (Get-Content $job.Config -Raw -Encoding UTF8 | ConvertFrom-Json).run_name }
        $path = Join-Path $Root "reports\eval_${tag}_c32.json"
        if (Test-Path $path) {
            $result = Get-Content $path -Raw | ConvertFrom-Json
            Write-Log ("SUMMARY {0}: val_loss {1} val_ppl {2}" -f $tag, $result.val_loss, $result.val_ppl)
        } else {
            Write-Log "SUMMARY ${tag}: no eval json"
        }
    }
} finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}
