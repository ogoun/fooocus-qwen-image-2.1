#!/usr/bin/env bash
# Builds the environment: .venv, torch with CUDA, the other dependencies, a readiness check.
# The order of the steps matters; see the comments in install.ps1.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$root/.venv"
python="$venv/bin/python"
torch_index="https://download.pytorch.org/whl/cu128"

step() { printf '\n\033[36m[%s/8] %s\033[0m\n' "$1" "$2"; }

cd "$root"

if [ "${1:-}" = "--recreate" ] && [ -d "$venv" ]; then
    echo "Removing the previous environment..."
    rm -rf "$venv"
fi

step 1 "Creating the environment and installing torch with CUDA"
if [ ! -x "$python" ]; then
    python3.12 -m venv "$venv" 2>/dev/null || python3.13 -m venv "$venv" 2>/dev/null || python3 -m venv "$venv"
fi
"$python" -m pip install --upgrade pip setuptools wheel
"$python" -m pip install torch torchvision --index-url "$torch_index"

step 2 "Installing the other dependencies"
"$python" -m pip install -r "$root/requirements.txt"

step 3 "Restoring the CUDA build of torch if it was replaced"
version="$("$python" -c 'import torch, sys; sys.stdout.write(torch.__version__)')"
case "$version" in
    *cu*) echo "  in place: $version" ;;
    *)    echo "  found $version, reinstalling"
          "$python" -m pip install --force-reinstall torch torchvision --index-url "$torch_index" ;;
esac

step 4 "Choosing weight precision and SageAttention"
"$python" -m fooocus_qwen --setup-performance

step 5 "Checking the model weights and the pose recognition weights"
"$python" -m fooocus_qwen --fetch-model

step 6 "Setting up the language model for prompt AI boost"
"$python" -m fooocus_qwen --setup-llm

step 7 "Video engine for the Video tab (optional, ~40-55 GB)"
"$python" -m fooocus_qwen --setup-video

step 8 "Checking readiness"
"$python" -m fooocus_qwen --selftest

printf '\n\033[32mDone. To start:\033[0m\n    ./run.sh\n'
