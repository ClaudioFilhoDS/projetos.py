# Remove o Presence Guard do Windows (a config e o cadastro do rosto ficam).
$App    = "presence-guard"
$Data   = Join-Path $env:LOCALAPPDATA $App
$Py     = Join-Path $Data "venv\Scripts\python.exe"
$Script = Join-Path $Data "presence_guard.py"
$Bin    = Join-Path $Data "bin"

if ((Test-Path $Py) -and (Test-Path $Script)) { & $Py $Script off | Out-Null }
Remove-Item (Join-Path ([Environment]::GetFolderPath("Startup")) "Presence Guard.lnk") -ErrorAction SilentlyContinue
Remove-Item (Join-Path ([Environment]::GetFolderPath("Programs")) "Presence Guard (ligar-desligar).lnk") -ErrorAction SilentlyContinue
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath) {
    $Kept = ($UserPath -split ";") | Where-Object { $_ -and ($_ -ne $Bin) }
    [Environment]::SetEnvironmentVariable("Path", ($Kept -join ";"), "User")
}
Start-Sleep -Seconds 1
Remove-Item $Data -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Removido. A config e o cadastro do rosto ficaram em $env:APPDATA\presence-guard (apague se quiser)."
