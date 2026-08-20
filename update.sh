#!/usr/bin/env sh

CURRENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

echo "***** Updating MoneyPrinter Custom *****"

git -C "$CURRENT_DIR" pull --rebase
if [ $? -ne 0 ]; then
  echo "***** Git pull failed. Commit or stash local changes first. *****"
  exit 1
fi

if [ -x "$CURRENT_DIR/.venv/bin/python" ]; then
  echo "***** Refreshing project virtual environment *****"
  "$CURRENT_DIR/.venv/bin/python" -m pip install -r requirements.txt
elif command -v uv >/dev/null 2>&1; then
  echo "***** Refreshing dependencies with uv *****"
  uv sync --frozen
else
  echo "***** Neither project Python nor uv found. Skipping dependency update. *****"
  echo "***** Install Python 3.11+, then run: pip install -r requirements.txt *****"
fi

echo "***** Update finished. Launch with webui.sh *****"