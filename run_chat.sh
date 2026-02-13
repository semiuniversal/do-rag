#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "${SCRIPT_DIR}/.venv" ]; then
    "${SCRIPT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/chat_client.py"
else
    python3 "${SCRIPT_DIR}/chat_client.py"
fi
