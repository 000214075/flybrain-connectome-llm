<#
Control for the automatic loop that keeps this project's standing request running.

The loop has two engines (see autoloop/README.md):
  1. a global Stop hook  -- re-injects autoloop/prompt.txt at the end of each turn;
  2. a scheduler task    -- fires after the hook's per-turn cap is exhausted (grok
                            stops consulting Stop hooks after 8 continuations in one
                            turn), and restarts the cycle.

This script turns both on and off and reports what is actually installed, so
"is it running?" is answered by inspection rather than by memory.

    powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Status
    powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Disable
    powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Enable
#>
[CmdletBinding()]
param(
    [switch]$Status,
    [switch]$Enable,
    [switch]$Disable,
    [switch]$ResetCounter
)

$ErrorActionPreference = 'Stop'

$Project     = 'C:\Users\heyiy\Desktop\cangying'
$Root        = Join-Path $Project 'autoloop'
$DisableFile = Join-Path $Root 'DISABLE'
$StateFile   = Join-Path $Root 'state.json'
$PromptFile  = Join-Path $Root 'prompt.txt'
$LogFile     = Join-Path $Root 'inject.log'
$HookFile    = Join-Path $env:GROK_HOME 'hooks\flybrain-autoloop.json'

function Get-Rounds {
    if (-not (Test-Path -LiteralPath $StateFile)) { return 0 }
    try {
        return [int](([System.IO.File]::ReadAllText($StateFile, [System.Text.Encoding]::UTF8) | ConvertFrom-Json).rounds)
    } catch { return 0 }
}

function Show-Status {
    Write-Output "project        : $Project"
    Write-Output "prompt         : $(if (Test-Path $PromptFile) { "$PromptFile ($((Get-Item $PromptFile).Length) bytes, last written $((Get-Item $PromptFile).LastWriteTime))" } else { 'MISSING' })"
    Write-Output "hook installed : $(if (Test-Path $HookFile) { $HookFile } else { "MISSING ($HookFile)" })"
    Write-Output "switch         : $(if (Test-Path $DisableFile) { 'DISABLED (autoloop/DISABLE exists)' } else { 'ENABLED' })"
    Write-Output "rounds so far  : $(Get-Rounds)"
    if (Test-Path $LogFile) {
        Write-Output "last log lines :"
        Get-Content -LiteralPath $LogFile -Tail 5 | ForEach-Object { "                 $_" }
    } else {
        Write-Output "last log lines : (no inject.log yet -- the hook has never fired)"
    }
    Write-Output ""
    Write-Output "note: the scheduler half lives in the grok session, not on disk."
    Write-Output "      check it with the /loop list, or scheduler_list."
}

if ($Status -or (-not ($Enable -or $Disable -or $ResetCounter))) { Show-Status; exit 0 }

if ($Enable) {
    if (Test-Path -LiteralPath $DisableFile) { Remove-Item -LiteralPath $DisableFile -Force }
    Write-Output "loop ENABLED (autoloop/DISABLE removed)."
    Write-Output "The next turn to finish in this project will be continued automatically."
}

if ($Disable) {
    if (-not (Test-Path -LiteralPath $Root)) { New-Item -ItemType Directory -Path $Root | Out-Null }
    Set-Content -LiteralPath $DisableFile -Encoding UTF8 -Value (
        "Put here by scripts\autoloop_ctl.ps1 -Disable at " + (Get-Date).ToString('s') + ".`r`n" +
        "While this file exists the Stop hook allows every turn to end normally.`r`n" +
        "Delete it (or run -Enable) to start the loop again.`r`n"
    )
    Write-Output "loop DISABLED (wrote $DisableFile)."
    Write-Output "The scheduler half, if any, still fires; remove it with scheduler_delete."
}

if ($ResetCounter) {
    [System.IO.File]::WriteAllText(
        $StateFile, '{ "rounds": 0, "last_inject": null, "last_prompt_id": null }',
        (New-Object System.Text.UTF8Encoding $false)
    )
    Write-Output "round counter reset to 0."
}
