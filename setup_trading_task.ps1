$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $projectRoot "server\unified_master.py"
$launcher = Join-Path $projectRoot "start_trading_session_hidden.vbs"
$envFile = Join-Path $projectRoot ".env"

if (-not (Test-Path $runner)) {
    throw "Unified supervisor not found: $runner"
}
if (-not (Test-Path $envFile)) {
    throw "Missing .env. Copy .env.example to .env and configure Alpaca paper credentials."
}

if (-not (Test-Path $launcher)) {
    throw "Hidden launcher not found: $launcher"
}

$wscript = Join-Path $env:SystemRoot "System32\wscript.exe"
$taskName = "ApexQuant-Paper-Trading"
$userId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction `
    -Execute $wscript `
    -Argument "`"$launcher`"" `
    -WorkingDirectory $projectRoot
$triggers = @(
    (New-ScheduledTaskTrigger -AtStartup),
    (New-ScheduledTaskTrigger -AtLogOn -User $userId),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At "2:00PM")
)
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $triggers `
    -Principal $principal `
    -Settings $settings `
    -Description "Starts the ApexQuant unified paper-only supervisor; the runner waits for Alpaca US market hours." `
    -Force | Out-Null

Write-Host "Registered $taskName for startup, user logon, and weekdays at 2:00 PM local Windows time."
Write-Host "The task runs as the current user without storing a password; the user must be logged in."
Write-Host "The scheduled action launches the batch file through a hidden Windows Script Host."
Write-Host "Project: $projectRoot"
Write-Host "Verify the PC Windows time zone is Asia/Jerusalem for the intended premarket schedule."
