# Instala o Presence Guard no Windows 11 (só para o seu usuário, sem administrador).
$ErrorActionPreference = "Stop"
$App    = "presence-guard"
$Src    = Split-Path -Parent $MyInvocation.MyCommand.Path
$Data   = Join-Path $env:LOCALAPPDATA $App
$Conf   = Join-Path $env:APPDATA $App
$Venv   = Join-Path $Data "venv"
$Py     = Join-Path $Venv "Scripts\python.exe"
$PyW    = Join-Path $Venv "Scripts\pythonw.exe"
$Script = Join-Path $Data "presence_guard.py"
$Bin    = Join-Path $Data "bin"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

function Find-Python {
    try {
        $p = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p -and (Test-Path $p)) { return $p }
    } catch {}
    foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
                     "$env:ProgramFiles\Python312\python.exe")) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

Step "Procurando o Python 3.12"
$Base = Find-Python
if (-not $Base) {
    Step "Instalando o Python 3.12 (winget, só para o seu usuário)"
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    $Base = Find-Python
    if (-not $Base) { throw "Não achei o Python 3.12 depois de instalar. Instale pelo python.org e rode de novo." }
}
Write-Host "Python: $Base"

if ((Test-Path $Py) -and (Test-Path $Script)) {
    Step "Parando a versão que está rodando"
    & $Py $Script off | Out-Null
}

Step "Criando o ambiente Python (MediaPipe, OpenCV...) - pode levar alguns minutos"
New-Item -ItemType Directory -Force -Path $Data, $Conf, $Bin | Out-Null
if (-not (Test-Path $Py)) { & $Base -m venv $Venv }
& $Py -m pip install --quiet --upgrade pip
& $Py -m pip install --quiet mediapipe mss
# API de câmera do Windows (WinRT): é por ela que dá para ler a câmera infravermelha
& $Py -m pip install --quiet winrt-runtime winrt-Windows.Foundation winrt-Windows.Foundation.Collections `
    winrt-Windows.Media winrt-Windows.Media.Capture winrt-Windows.Media.Capture.Frames `
    winrt-Windows.Media.Devices winrt-Windows.Media.Devices.Core winrt-Windows.Media.MediaProperties `
    winrt-Windows.Graphics.Imaging winrt-Windows.Graphics.DirectX.Direct3D11 winrt-Windows.Storage.Streams `
    winrt-Windows.Devices.Enumeration winrt-Windows.Perception.Spatial
if ($LASTEXITCODE -ne 0) { throw "A instalação dos pacotes Python falhou (veja a mensagem acima)." }

Step "Copiando os arquivos"
Copy-Item (Join-Path $Src "presence_guard.py") $Script -Force
$Cfg = Join-Path $Conf "config.ini"
if ((Test-Path $Cfg) -and -not (Select-String -Path $Cfg -Pattern '^\[borrao\]' -Quiet)) {
    Copy-Item $Cfg "$Cfg.anterior" -Force
    Write-Host "Config da versão anterior guardada em $Cfg.anterior"
    Remove-Item $Cfg
}
if (-not (Test-Path $Cfg)) { Copy-Item (Join-Path $Src "config.ini") $Cfg }

Step "Baixando os modelos de rosto (~43 MB, só na primeira vez)"
& $Py $Script download-models
if ($LASTEXITCODE -ne 0) { throw "Não consegui baixar os modelos (sem internet?)." }

Step "Comando presence-guard (funciona nos terminais abertos depois daqui)"
Set-Content -Path (Join-Path $Bin "presence-guard.cmd") -Encoding ASCII -Value `
    '@"%LOCALAPPDATA%\presence-guard\venv\Scripts\python.exe" "%LOCALAPPDATA%\presence-guard\presence_guard.py" %*'
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $UserPath) { $UserPath = "" }
if (-not (($UserPath -split ";") -contains $Bin)) {
    $NewPath = (@($UserPath.TrimEnd(";"), $Bin) | Where-Object { $_ }) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
}

Step "Atalhos: iniciar junto com o Windows e Ctrl+Alt+P para ligar/desligar"
$Shell = New-Object -ComObject WScript.Shell
$Link = $Shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Startup")) "Presence Guard.lnk"))
$Link.TargetPath = $PyW
$Link.Arguments = "`"$Script`" supervise"
$Link.WorkingDirectory = $Data
$Link.Description = "Presence Guard"
$Link.Save()
$Link = $Shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Programs")) "Presence Guard (ligar-desligar).lnk"))
$Link.TargetPath = $PyW
$Link.Arguments = "`"$Script`" toggle"
$Link.WorkingDirectory = $Data
$Link.Hotkey = "CTRL+ALT+P"
$Link.Description = "Liga/desliga o Presence Guard"
$Link.Save()

Step "Procurando as câmeras (a luz da câmera pode piscar)"
& $Py $Script cameras

Step "Ligando"
& $Py $Script on

Write-Host "`nInstalado!" -ForegroundColor Green
$Answer = Read-Host "Calibrar o olhar agora? (25 s) [S/n]"
if ($Answer -notmatch '^[nN]') { & $Py $Script calibrate }
$Answer = Read-Host "Cadastrar seu rosto com e sem óculos agora? (1 min) [S/n]"
if ($Answer -notmatch '^[nN]') { & $Py $Script cadastrar }
Write-Host ""
Write-Host "Pronto! Ctrl+Alt+P liga/desliga. Em um terminal NOVO:"
Write-Host "  presence-guard off | on | toggle | preview | cameras | status | log"
