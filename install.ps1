<#
.SYNOPSIS
    Сборка окружения: .venv, torch с CUDA, остальные зависимости, проверка готовности.

.DESCRIPTION
    Порядок шагов важен:

    1. torch ставится ПЕРВЫМ и с индекса pytorch. Поставленный следом за
       остальными, он приедет с PyPI — без CUDA, и всё пойдёт на процессоре.
    2. Остальные зависимости из requirements.txt.
    3. torch закрепляется ещё раз: сторонние пакеты способны подменить
       CUDA-сборку на обычную.
    4. Веса модели: тридцать три гигабайта, качаются только если их нет.
    5. Адрес языковой модели для AI-буста промтов — по желанию.
    6. Самопроверка: «установилось» должно означать «запустится».

    Шаги 4 и 5 делает сам пакет (`--fetch-model`, `--setup-llm`): то же
    самое, написанное дважды на двух языках оболочки, разъезжается.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -Recreate
#>

[CmdletBinding()]
param([switch] $Recreate)

$ErrorActionPreference = 'Stop'
# Консоль Windows по умолчанию не UTF-8, и русский текст выводится мусором.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$torchIndex = 'https://download.pytorch.org/whl/cu128'

function Step($number, $text) {
    Write-Host ''
    Write-Host "[$number/6] $text" -ForegroundColor Cyan
}

Push-Location $root
try {
    if ($Recreate -and (Test-Path $venv)) {
        Write-Host 'Удаляю прежнее окружение…' -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venv
    }

    Step 1 'Создаю окружение и ставлю torch с поддержкой CUDA'
    if (-not (Test-Path $python)) {
        py -3.12 -m venv $venv
        if ($LASTEXITCODE -ne 0) {
            Write-Host '  Python 3.12 не найден, пробую 3.13' -ForegroundColor Yellow
            py -3.13 -m venv $venv
        }
        if (-not (Test-Path $python)) { throw 'Не удалось создать .venv' }
    }
    & $python -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось обновить pip' }

    & $python -m pip install torch torchvision --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось поставить torch с CUDA' }

    Step 2 'Ставлю остальные зависимости'
    & $python -m pip install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось поставить зависимости' }

    Step 3 'Возвращаю CUDA-сборку torch, если её заменили'
    $version = & $python -c "import torch, sys; sys.stdout.write(torch.__version__)"
    if ($version -notlike '*cu*') {
        Write-Host "  сейчас стоит $version — переустанавливаю" -ForegroundColor Yellow
        & $python -m pip install --force-reinstall torch torchvision --index-url $torchIndex
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось вернуть torch с CUDA' }
    } else {
        Write-Host "  всё на месте: $version"
    }

    Step 4 'Проверяю веса модели'
    & $python -m fooocus_qwen --fetch-model
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось получить веса модели' }

    Step 5 'Настраиваю языковую модель для AI-буста промтов'
    & $python -m fooocus_qwen --setup-llm

    Step 6 'Проверяю готовность'
    & $python -m fooocus_qwen --selftest
    if ($LASTEXITCODE -ne 0) { throw 'Самопроверка не пройдена' }

    Write-Host ''
    Write-Host 'Готово. Запуск:' -ForegroundColor Green
    Write-Host '    .\run.ps1' -ForegroundColor Cyan
} finally {
    Pop-Location
}
