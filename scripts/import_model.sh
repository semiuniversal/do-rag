#!/bin/bash
# import_model.sh — Import the Local File Expert model preset into Open WebUI
# Uses admin credentials (no API key needed). Set OPENWEBUI_ADMIN_EMAIL and OPENWEBUI_ADMIN_PASSWORD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_JSON="$PROJECT_ROOT/open_webui/local-file-expert.json"
WEBUI_URL="${OPENWEBUI_URL:-http://localhost:3000}"
ADMIN_EMAIL="${OPENWEBUI_ADMIN_EMAIL:-admin@local}"

if [ -z "$OPENWEBUI_ADMIN_PASSWORD" ]; then
  echo "Error: OPENWEBUI_ADMIN_PASSWORD is not set."
  echo "Use the same admin password you set for Open WebUI (or OPENWEBUI_ADMIN_PASSWORD in run.sh)."
  echo "  OPENWEBUI_ADMIN_PASSWORD=your_password ./scripts/import_model.sh"
  exit 1
fi

if [ ! -f "$MODEL_JSON" ]; then
  echo "Error: Model JSON not found at $MODEL_JSON"
  exit 1
fi

echo "Waiting for Open WebUI at $WEBUI_URL..."
for i in $(seq 1 60); do
  if curl -sf "$WEBUI_URL/health" > /dev/null 2>&1; then
    echo "Open WebUI is ready. Logging in and importing..."
    TOKEN=$(curl -sf -X POST "$WEBUI_URL/api/v1/auths/signin" \
      -H "Content-Type: application/json" \
      -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$OPENWEBUI_ADMIN_PASSWORD\"}" 2>/dev/null | \
      python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('token',d.get('access_token','')))" 2>/dev/null)
    if [ -z "$TOKEN" ]; then
      echo "Login failed. Check email and password."
      exit 1
    fi
    if curl -sf -X POST "$WEBUI_URL/api/v1/models/import" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d @"$MODEL_JSON" > /dev/null 2>&1; then
      echo "Model imported successfully."
      exit 0
    else
      echo "Import failed (model may already exist)."
      exit 1
    fi
  fi
  sleep 2
done

echo "Timeout: Open WebUI did not become ready."
exit 1
