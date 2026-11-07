param(
    [Parameter(Mandatory = $true)]
    [string]$PrivateKeyPath,

    [Parameter(Mandatory = $true)]
    [string]$VpsHost,

    [string]$VpsUser = "ubuntu"
)

$ErrorActionPreference = "Stop"

function Invoke-Icacls {
    param([string[]]$Arguments)

    & icacls.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "icacls failed with exit code $LASTEXITCODE."
    }
}

if ($VpsHost -match "[<>\s]") {
    throw "VpsHost must be the actual public IP address or DNS name, without brackets or spaces."
}

if ($VpsUser -notmatch "^[A-Za-z0-9._-]+$") {
    throw "VpsUser contains unsupported characters."
}

$sshDirectory = Join-Path $env:USERPROFILE ".ssh"
if (-not (Test-Path -LiteralPath $sshDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $sshDirectory | Out-Null
}

$resolvedKeyPath = [System.IO.Path]::GetFullPath($PrivateKeyPath)
$resolvedSshDirectory = [System.IO.Path]::GetFullPath($sshDirectory).TrimEnd("\") + "\"
if (-not $resolvedKeyPath.StartsWith(
        $resolvedSshDirectory,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "Store the private key under $sshDirectory, not in the project directory."
}

if (-not (Test-Path -LiteralPath $resolvedKeyPath -PathType Leaf)) {
    throw "Private key file not found: $resolvedKeyPath"
}

$publicKeyPath = "$resolvedKeyPath.pub"
$privatePublicKey = & ssh-keygen.exe -y -f $resolvedKeyPath
if ($LASTEXITCODE -ne 0 -or -not $privatePublicKey) {
    throw "Unable to read the private key. Check that it is a valid OpenSSH key and enter its passphrase if requested."
}

if (Test-Path -LiteralPath $publicKeyPath -PathType Leaf) {
    $existingPublicKey = (Get-Content -LiteralPath $publicKeyPath -Raw).Trim()
    $existingFields = $existingPublicKey -split "\s+"
    $derivedFields = ($privatePublicKey -join "").Trim() -split "\s+"
    if ($existingFields.Count -lt 2 -or $derivedFields.Count -lt 2 -or
        $existingFields[0] -ne $derivedFields[0] -or
        $existingFields[1] -ne $derivedFields[1]) {
        throw "The existing public key does not match the selected private key: $publicKeyPath"
    }
}
else {
    Set-Content -LiteralPath $publicKeyPath -Value ($privatePublicKey -join "") -Encoding ascii
}

$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
Invoke-Icacls -Arguments @($resolvedKeyPath, "/setowner", $identity)
Invoke-Icacls -Arguments @($resolvedKeyPath, "/inheritance:r")
Invoke-Icacls -Arguments @($resolvedKeyPath, "/remove:g", "*S-1-5-32-544")
Invoke-Icacls -Arguments @(
    $resolvedKeyPath,
    "/grant:r",
    "${identity}:(R)",
    "*S-1-5-18:(F)"
)

Write-Host "Private key ACL:"
& icacls.exe $resolvedKeyPath
if ($LASTEXITCODE -ne 0) {
    throw "Unable to verify private key permissions."
}

Write-Host "Public key fingerprint:"
$fingerprint = & ssh-keygen.exe -lf $publicKeyPath
if ($LASTEXITCODE -ne 0) {
    throw "Unable to verify public key fingerprint."
}
Write-Host $fingerprint
if ($fingerprint -notmatch "\(ED25519\)") {
    throw "Phase 0 requires an Ed25519 key. Generate a new one with ssh-keygen -t ed25519."
}

Write-Host ""
Write-Host "Ensure this public key is installed in $VpsUser's ~/.ssh/authorized_keys on the VPS:"
Write-Host $publicKeyPath
Write-Host "Display the public key for installation (public material only):"
Write-Host "Get-Content -Raw `"$publicKeyPath`""
Write-Host ""
Write-Host "For passphrase-protected keys, start the Windows ssh-agent and add the key once:"
Write-Host "  Set-Service ssh-agent -StartupType Automatic  # run elevated if required"
Write-Host "  Start-Service ssh-agent"
Write-Host "  ssh-add `"$resolvedKeyPath`""
Write-Host ""

$target = "${VpsUser}@${VpsHost}"
& ssh.exe -o BatchMode=yes -o IdentitiesOnly=yes -i $resolvedKeyPath $target "printf 'SSH key authentication OK\n'"
if ($LASTEXITCODE -ne 0) {
    throw "SSH key login failed. Verify the public key is authorized on the VPS and the host fingerprint is trusted."
}
