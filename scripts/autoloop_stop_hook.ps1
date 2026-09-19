<#
Stop hook: keep the fly-brain project's standing request running.

Registered as a *global* hook (GROK_HOME\hooks\*.json) rather than a project one,
because this repo is not in trusted_folders.toml -- a hook under
<project>/.grok/hooks/ is silently skipped until the folder is trusted, and a
silently skipped hook is worse than no hook. To stay safe as a global hook, this
script acts only when the session's workspace is exactly $Project.

Contract (grok user-guide 10-hooks.md):
  stdin  : JSON envelope, common fields include hookEventName, cwd, workspaceRoot,
           reason, promptId, and event-specific ones (Stop carries `reason`,
           `backgroundTasks`, `sessionCrons`, `stopHookActive`).
  stdout : `{"decision":"block","reason":"..."}` keeps the agent working, with the
           reason fed back as a user message. Anything else (or no output) allows
           the stop. grok stops consulting hooks after 8 continuations in one turn,
           so this cannot spin forever; the scheduler restarts the cycle after that.

Every allow path is logged with its reason, so "why did it not fire" is always
answerable from autoloop/inject.log instead of guessed at.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'SilentlyContinue'

$Project     = if ($env:AUTOLOOP_PROJECT) { $env:AUTOLOOP_PROJECT } else { 'C:\Users\heyiy\Desktop\cangying' }
$Root        = if ($env:AUTOLOOP_ROOT) { $env:AUTOLOOP_ROOT } else { Join-Path $Project 'autoloop' }
$PromptFile  = Join-Path $Root 'prompt.txt'
$DisableFile = Join-Path $Root 'DISABLE'
$StateFile   = Join-Path $Root 'state.json'
$LogFile     = Join-Path $Root 'inject.log'

function Write-Log([string]$Message) {
    try {
        $stamp = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss')
        $line = "$stamp pid=$PID $Message"
        # Append through a BOM-less encoder: Add-Content -Encoding UTF8 writes a BOM
        # at the start of a fresh file, which every plain-UTF-8 reader then trips on.
        $bytes = [System.Text.Encoding]::UTF8.GetBytes("$line`r`n")
        $stream = [System.IO.File]::Open($LogFile, 'Append', 'Write', 'ReadWrite')
        try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
    } catch { }
}

function Exit-Allow([string]$Why) {
    Write-Log "allow: $Why"
    exit 0
}

# Explicit UTF-8 read: Windows PowerShell 5.1's Get-Content reads a BOM-less UTF-8
# file as ANSI, which would turn the Chinese prompt into mojibake.
function Read-Utf8([string]$Path) {
    # File.ReadAllText strips a leading BOM, unlike a naive byte decode.
    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

# Set-Content -Encoding UTF8 on Windows PowerShell 5.1 writes a BOM, which then breaks
# any reader that decodes the file as plain UTF-8 (this project's own test caught it).
# Writing through a BOM-less encoder keeps the state file valid JSON for everyone.
function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding $false))
}

# Write the decision as pure ASCII. Two separate hazards live here, both measured on
# this host:
#   * Windows PowerShell 5.1 reads a .ps1 file with no BOM as ANSI, so a Chinese
#     literal *inside a script* is already corrupted before it runs. That is why the
#     prompt is never inlined here and is read from prompt.txt as explicit UTF-8.
#   * Writing the decision to the standard output stream put a UTF-8 BOM in front of
#     it, and grok then cannot parse the JSON -- a hook that silently never fires.
# Escaping every non-ASCII code point to \uXXXX removes both: the bytes on the wire
# are ASCII, so no console code page can touch them, and JSON decodes the escapes
# back into the original Chinese on the other side.
function ConvertTo-AsciiJson([string]$Text) {
    $sb = New-Object System.Text.StringBuilder
    foreach ($ch in $Text.ToCharArray()) {
        $code = [int]$ch
        if ($code -gt 126 -or $code -lt 32) {
            [void]$sb.AppendFormat('\u{0:x4}', $code)
        } else {
            [void]$sb.Append($ch)
        }
    }
    return $sb.ToString()
}

function Write-Decision([string]$Json) {
    # No BOM: a preamble here would break the JSON parse on the other side.
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
    Write-Output (ConvertTo-AsciiJson $Json)
}

$raw = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($raw)) { Exit-Allow 'empty stdin' }

try { $envelope = $raw | ConvertFrom-Json } catch { Exit-Allow 'stdin was not JSON' }
if ($null -eq $envelope) { Exit-Allow 'stdin was JSON null' }

# --- guards: every one of these means "let the turn end normally" -------------
if ($envelope.hookEventName -and $envelope.hookEventName -ne 'stop') {
    Exit-Allow "event is $($envelope.hookEventName), not stop"
}
# A session-end fire (channel_closed / shutdown) has no turn left to continue.
if ($envelope.reason -ne 'end_turn') {
    Exit-Allow "reason is '$($envelope.reason)', not end_turn"
}
if (Test-Path -LiteralPath $DisableFile) { Exit-Allow 'autoloop/DISABLE exists' }
if (-not (Test-Path -LiteralPath $PromptFile)) { Exit-Allow 'autoloop/prompt.txt is missing' }

# Compare workspaces with separators and trailing slashes normalised. The envelope's
# path style is not ours to choose: the documented examples are POSIX-style
# (`/Users/you/project`), so a forward-slash spelling of a Windows path is entirely
# plausible, and a string compare that misses it makes this hook silently never fire
# while looking perfectly installed.
function Normalize-Path([string]$Value) {
    return $Value.Replace('/', '\').TrimEnd('\')
}

$where = if ($envelope.workspaceRoot) { $envelope.workspaceRoot } else { $envelope.cwd }
if ([string]::IsNullOrWhiteSpace($where)) { Exit-Allow 'envelope carried no cwd' }
if ((Normalize-Path $where) -ine (Normalize-Path $Project)) {
    Exit-Allow "workspace is $where, not the fly-brain project"
}

# Something is already running: its completion will wake the session by itself, so
# injecting here would only stack a second round on top of work in flight.
$inflight = 0
if ($envelope.backgroundTasks) { $inflight = @($envelope.backgroundTasks).Count }
if ($inflight -gt 0) { Exit-Allow "$inflight background task(s) in flight" }

# ...and that guard only sees grok's own background tasks. A training run started by
# hand, or by another session, is invisible to it. Measured cost of getting this wrong:
# a 40-minute run sharing the GPU with a web server went from 218 tok/s to 51, and a
# round nearly published "the shuffled wiring is 3.3x slower" on the strength of it.
#
# `-QuietMinutes 0` on purpose: that disables the check's *recency* signal and leaves
# only the running-job signal. This hook fires at the end of a turn, and the turn that
# just ended is exactly what wrote those recent files -- with the default 5-minute
# window the hook would report BUSY after every single round and the fast engine would
# never fire at all, silently. The scheduler keeps the default window; as a once-an-hour
# backstop, being conservative there costs at most one firing.
#
# The check fails FREE on error, so a broken check can never wedge the loop.
$busyCheck = Join-Path $Project 'scripts\autoloop_busy_check.ps1'
if (Test-Path -LiteralPath $busyCheck) {
    try {
        & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $busyCheck -QuietMinutes 0 *> $null
        if ($LASTEXITCODE -eq 3) { Exit-Allow 'a project job is running (autoloop_busy_check)' }
    } catch {
        Write-Log "busy check failed (ignored, failing free): $_"
    }
}

# --- inject the next round ----------------------------------------------------
$body = Read-Utf8 $PromptFile
if ([string]::IsNullOrWhiteSpace($body)) { Exit-Allow 'prompt.txt is empty' }

$round = 0
if (Test-Path -LiteralPath $StateFile) {
    try { $round = [int]((Read-Utf8 $StateFile | ConvertFrom-Json).rounds) } catch { $round = 0 }
}
$round = $round + 1
$stamp = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss')
try {
    Write-Utf8NoBom $StateFile (
        '{ "rounds": ' + $round + ', "last_inject": "' + $stamp + '", "last_prompt_id": "' + $envelope.promptId + '" }'
    )
} catch { }

$reason = $body.Replace('{{ROUND}}', "$round")
$json = @{ decision = 'block'; reason = $reason } | ConvertTo-Json -Depth 6 -Compress
Write-Decision $json
Write-Log "inject: round $round (promptId $($envelope.promptId))"
exit 0
