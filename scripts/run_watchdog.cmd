@echo off
REM Launch the training watchdog from Task Scheduler or by hand.
REM
REM Why a .cmd wrapper instead of pointing the scheduled task straight at
REM powershell.exe -File:
REM
REM   * Task Scheduler starts a process with an arbitrary working directory
REM     (C:\Windows\System32) and no console. Pinning the directory here removes
REM     one variable from a launch that was already failing for reasons the
REM     scheduler would not report.
REM   * A scheduled task that hangs is silent: the scheduler just says the task is
REM     "running", which is indistinguishable from working. Redirecting both
REM     streams to a file means a failure leaves evidence.
REM
REM THE CONFIG IS NOW AN ARGUMENT. It used to be hardcoded to
REM configs\train_connectome_long2.json, and that hardcoded target cost 20 minutes
REM of GPU: after that run finished and was promoted, its output directory was
REM renamed to checkpoints\flybrain-connectome, so the config's `out_dir` no longer
REM existed. `resume: auto` then found no checkpoint and silently started a fresh
REM 18000-step run from zero -- reproducing a five-hour result that was already
REM measured and promoted. A launch script must not carry a hidden, stale target:
REM passing the config in makes it obvious which run is being started, and makes it
REM impossible to start the wrong one by not looking.
REM
REM Usage:
REM   scripts\run_watchdog.cmd <config.json>
REM
REM Before running, check what it will do:
REM   findstr out_dir <config.json>          -- which directory it writes to
REM   dir checkpoints\<that directory>        -- does a checkpoint exist to resume from
REM A missing checkpoint directory means the run starts from step 0.

setlocal
set "ROOT=C:\Users\heyiy\Desktop\cangying"
set "LOG=%ROOT%\reports\wd_task.log"

if "%~1"=="" (
  echo ERROR: no config given.
  echo Usage: run_watchdog.cmd ^<config.json^>
  echo Refusing to guess: a hardcoded default is what started a duplicate run before.
  exit /b 2
)

set "CONFIG=%~1"
if not exist "%CONFIG%" (
  if exist "%ROOT%\%CONFIG%" (set "CONFIG=%ROOT%\%CONFIG%") else (
    echo ERROR: config not found: %~1
    exit /b 2
  )
)

cd /d "%ROOT%"
echo [%DATE% %TIME%] watcher launcher starting with config=%CONFIG% >> "%LOG%"

"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" ^
  -NoProfile -ExecutionPolicy Bypass ^
  -File "%ROOT%\scripts\train_watchdog.ps1" ^
  -Config "%CONFIG%" >> "%LOG%" 2>&1

echo [%DATE% %TIME%] watcher launcher exited with %ERRORLEVEL% >> "%LOG%"
endlocal
