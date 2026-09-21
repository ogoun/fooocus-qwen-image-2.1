<#
.SYNOPSIS
    Запуск оболочки в окружении проекта.

.DESCRIPTION
    Приложение работает только в собственном .venv: в системном Python нет ни
    diffusers из git, ни torch с CUDA. Скрипт проверяет не факт установки, а
    работоспособность — окружение могло остаться от прерванной установки.

    Все аргументы передаются приложению как есть.

.EXAMPLE
    .\run.ps1

.EXAMPLE
    .\run.ps1 --lang en --port 7870
#>

[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Arguments)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host 'Окружение .venv не найдено.' -ForegroundColor Yellow
    Write-Host 'Создайте его один раз:' -ForegroundColor Yellow
    Write-Host '    .\install.ps1' -ForegroundColor Cyan
    exit 1
}

$check = & $python -c "import torch, sys; sys.stdout.write('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Окружение .venv повреждено: не импортируется torch.' -ForegroundColor Red
    Write-Host '    Remove-Item -Recurse -Force .venv; .\install.ps1' -ForegroundColor Cyan
    exit 1
}
if ($check -ne 'cuda') {
    Write-Host 'CUDA недоступна — генерация пойдёт на процессоре и займёт часы.' -ForegroundColor Yellow
    Write-Host 'Переустановите torch:' -ForegroundColor Yellow
    Write-Host '    .venv\Scripts\python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128' -ForegroundColor Cyan
}

Push-Location $root
try {
    & $python -m fooocus_qwen @Arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
