#!/usr/bin/env bash
# Сборка окружения: .venv, torch с CUDA, остальные зависимости, проверка готовности.
# Порядок шагов важен — см. комментарии в install.ps1.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$root/.venv"
python="$venv/bin/python"
torch_index="https://download.pytorch.org/whl/cu128"

step() { printf '\n\033[36m[%s/4] %s\033[0m\n' "$1" "$2"; }

cd "$root"

if [ "${1:-}" = "--recreate" ] && [ -d "$venv" ]; then
    echo "Удаляю прежнее окружение…"
    rm -rf "$venv"
fi

step 1 "Создаю окружение и ставлю torch с поддержкой CUDA"
if [ ! -x "$python" ]; then
    python3.12 -m venv "$venv" 2>/dev/null || python3 -m venv "$venv"
fi
"$python" -m pip install --upgrade pip setuptools wheel
"$python" -m pip install torch torchvision --index-url "$torch_index"

step 2 "Ставлю остальные зависимости"
"$python" -m pip install -r "$root/requirements.txt"

step 3 "Возвращаю CUDA-сборку torch, если её заменили"
version="$("$python" -c 'import torch, sys; sys.stdout.write(torch.__version__)')"
case "$version" in
    *cu*) echo "  всё на месте: $version" ;;
    *)    echo "  сейчас стоит $version — переустанавливаю"
          "$python" -m pip install --force-reinstall torch torchvision --index-url "$torch_index" ;;
esac

step 4 "Проверяю готовность"
"$python" -m fooocus_qwen --selftest

printf '\n\033[32mГотово. Запуск:\033[0m\n    ./run.sh\n'
