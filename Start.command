#!/bin/zsh
set -e
cd -- "${0:A:h}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Zuerst installieren: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt'
    exit 1
fi
exec .venv/bin/python llauncher.py "$@"
