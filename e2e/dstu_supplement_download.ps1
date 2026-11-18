param([Parameter(Mandatory=$true)][string]$Archive)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $taskRoot 'e2e/dstu_supplement/dist'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$sshArguments = @('-i','C:/Users/YiyangLiu/.ssh/crisp_server','-o','HostKeyAlias=118.195.219.207','-o','BatchMode=yes','-o','ConnectTimeout=15','ubuntu@146.56.225.29')
foreach ($name in @($Archive, "$Archive.json")) {
    $object = "cos://crisp/crisp/DSTU_SUPPLEMENT_20261118/dist/$name"
    $signedOutput = & ssh @sshArguments "/home/ubuntu/bin/coscli signurl '$object' --simple-output -t 3600"
    if ($LASTEXITCODE -ne 0) { throw "COS signing failed for $name" }
    $signedUrl = ($signedOutput | Where-Object { $_ -match '^https://' } | Select-Object -Last 1).Trim()
    $target = Join-Path $destination $name
    try {
        Invoke-WebRequest -Uri $signedUrl -OutFile "$target.part" -TimeoutSec 120
    } catch {
        throw "COS download failed for $name; partial file retained."
    }
    Move-Item -LiteralPath "$target.part" -Destination $target -Force
    Get-Item -LiteralPath $target | Select-Object Name,Length
}
$env:PYTHONUTF8='1'
& (Join-Path $taskRoot 'e2e/post_fg/dist/venv/Scripts/python.exe') -B (Join-Path $taskRoot 'e2e/dstu_supplement_collect.py') restore --archive (Join-Path $destination $Archive)
if ($LASTEXITCODE -ne 0) { throw 'Archive verification failed' }
