#!/usr/bin/env bash
# Запуск оболочки в окружении проекта. Все аргументы уходят приложению как есть.
#
# Браузер открывает само приложение, когда сервер готов отдавать страницу:
# скрипт момента готовности не знает, а Python с torch стартует секунды.
# Отключается ключом --no-open-browser.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$root/.venv/bin/python"

if [ ! -x "$python" ]; then
    echo "Окружение .venv не найдено. Создайте его один раз: ./install.sh" >&2
    exit 1
fi

if ! "$python" -c 'import torch' >/dev/null 2>&1; then
    echo "Окружение .venv повреждено: не импортируется torch." >&2
    echo "    rm -rf .venv && ./install.sh" >&2
    exit 1
fi

if [ "$("$python" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')" != "cuda" ]; then
    echo "CUDA недоступна — генерация пойдёт на процессоре и займёт часы." >&2
fi

cd "$root"
# Ключ первым: --no-open-browser из аргументов пользователя окажется после
# него и победит, argparse берёт последнее значение.
exec "$python" -m fooocus_qwen --open-browser "$@"
