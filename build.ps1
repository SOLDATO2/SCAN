# Script PowerShell para Windows — execute em PowerShell a partir da pasta do projeto: .\build.ps1
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Ativa venv (PowerShell)
$activatePs = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"

if (Test-Path $activatePs) {
    Write-Output "Ativando venv (PowerShell)..."
    . $activatePs
} else {
    Write-Error ".venv não encontrado. Crie-o com: python -m venv .venv"
    exit 1
}

Write-Output "Instalando/atualizando PyInstaller..."
pip install --upgrade pyinstaller

Write-Output "Executando PyInstaller..."
$pyArgs = @(
    "--clean"
    "--name=VideoInterpolation"
    "--windowed"
    "--add-data=assets;assets"
    "--add-data=adicionar_it.py;."
    "--add-data=model;model"
    "--add-data=dataset;dataset"
    "--add-data=trainer;trainer"
    "--hidden-import=adicionar_it"
    "--hidden-import=model.layers"
    "--hidden-import=dataset.frame_dataset"
    "--hidden-import=trainer.trainer"
    "--collect-all=torch"
    "--collect-all=torchvision"
    "--exclude-module=matplotlib"
    "--exclude-module=pandas"
    "interface.py"
)

& pyinstaller @pyArgs

if ($LASTEXITCODE -eq 0) {
    Write-Output "Build concluído! Executável em: dist\VideoInterpolation\"
} else {
    Write-Error "PyInstaller retornou código $LASTEXITCODE"
    exit
}