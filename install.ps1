<#
.SYNOPSIS
    Builds the environment: .venv, torch with CUDA, the other dependencies, a readiness check.

.DESCRIPTION
    The order of the steps matters:

    1. torch is installed FIRST, from the PyTorch index. Installed after the
       other packages, it would come from PyPI without CUDA, and everything
       would run on the CPU.
    2. The other dependencies from requirements.txt.
    3. torch is pinned once more: third-party packages can replace the CUDA
       build with a plain one.
    4. Performance: weight precision (bf16, INT8 or a GGUF variant; the
       default fits the detected video card, Q4_K_M on 6-8 GB) and
       SageAttention. The answer decides what the next step downloads.
    5. Model weights (about 33 GB for bf16, 26 GB for INT8, 23 GB for GGUF)
       and the pose recognition weights (DWPose, 350 MB); only missing files
       are downloaded. On cards under 20 GB the INT8 text encoder is built
       here as well (about 20 s).
    6. The language model address for prompt AI boost (optional).
    7. The video engine for the Video tab: FreeVideo with MiniMax H3, its own
       Python environment and ~40-55 GB of weights (optional, asks first).
    8. Self-test: "installed" must mean "will start".

    Steps 4-7 are done by the package itself (--setup-performance,
    --fetch-model, --setup-llm, --setup-video): the same logic written twice,
    in two shell languages, drifts apart.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -Recreate
#>

[CmdletBinding()]
param([switch] $Recreate)

$ErrorActionPreference = 'Stop'
# The Windows console is not UTF-8 by default; pip and the package may print
# non-ASCII text (paths, model names).
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$torchIndex = 'https://download.pytorch.org/whl/cu128'

function Step($number, $text) {
    Write-Host ''
    Write-Host "[$number/8] $text" -ForegroundColor Cyan
}

Push-Location $root
try {
    if ($Recreate -and (Test-Path $venv)) {
        Write-Host 'Removing the previous environment...' -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venv
    }

    Step 1 'Creating the environment and installing torch with CUDA'
    if (-not (Test-Path $python)) {
        py -3.12 -m venv $venv
        if ($LASTEXITCODE -ne 0) {
            Write-Host '  Python 3.12 not found, trying 3.13' -ForegroundColor Yellow
            py -3.13 -m venv $venv
        }
        if (-not (Test-Path $python)) { throw 'Could not create .venv' }
    }
    & $python -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip' }

    & $python -m pip install torch torchvision --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw 'Could not install torch with CUDA' }

    Step 2 'Installing the other dependencies'
    & $python -m pip install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the dependencies' }

    Step 3 'Restoring the CUDA build of torch if it was replaced'
    $version = & $python -c "import torch, sys; sys.stdout.write(torch.__version__)"
    if ($version -notlike '*cu*') {
        Write-Host "  found $version, reinstalling" -ForegroundColor Yellow
        & $python -m pip install --force-reinstall torch torchvision --index-url $torchIndex
        if ($LASTEXITCODE -ne 0) { throw 'Could not restore torch with CUDA' }
    } else {
        Write-Host "  in place: $version"
    }

    Step 4 'Choosing weight precision and SageAttention'
    & $python -m fooocus_qwen --setup-performance

    Step 5 'Checking the model weights and the pose recognition weights'
    & $python -m fooocus_qwen --fetch-model
    if ($LASTEXITCODE -ne 0) { throw 'Could not get the model weights' }

    Step 6 'Setting up the language model for prompt AI boost'
    & $python -m fooocus_qwen --setup-llm

    Step 7 'Video engine for the Video tab (optional, ~40-55 GB)'
    & $python -m fooocus_qwen --setup-video

    Step 8 'Checking readiness'
    & $python -m fooocus_qwen --selftest
    if ($LASTEXITCODE -ne 0) { throw 'Self-test failed' }

    Write-Host ''
    Write-Host 'Done. To start:' -ForegroundColor Green
    Write-Host '    .\run.ps1' -ForegroundColor Cyan
} finally {
    Pop-Location
}
