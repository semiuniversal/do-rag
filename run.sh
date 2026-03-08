#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load credentials from .env (created by first-run setup)
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
fi

# First-run setup: prompt for missing credentials and write .env
if ! "$SCRIPT_DIR/scripts/setup_env.sh"; then
  echo "Setup failed. Please fix the errors above and try again."
  exit 1
fi

# Reload .env in case setup just wrote it
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Stop any running stack first (clean slate; aborts indexing if running)
./stop.sh > /dev/null 2>&1
sleep 2

echo "=== Starting do-rag Services ==="

# 1. Start Ollama (Force Restart to ensure port binding)
echo "[1/7] Restarting Ollama..."
./run_ollama.sh stop > /dev/null 2>&1
sleep 1
./run_ollama.sh start

# 2. Pull required models if missing
echo "[2/7] Checking models..."
EMBEDDING_MODEL=$(python3 -c "import config; print(config.EMBEDDING_MODEL)" 2>/dev/null)
LLM_MODEL=$(python3 -c "import config; print(config.LLM_MODEL)" 2>/dev/null)

for model in "$EMBEDDING_MODEL" "$LLM_MODEL"; do
    if [ -n "$model" ] && ! ollama list 2>/dev/null | grep -q "$model"; then
        echo "  Pulling $model..."
        ollama pull "$model"
    else
        echo "  $model ✓"
    fi
done

# 3. Start Qdrant (Podman/Docker)
echo "[3/7] Starting Qdrant..."
./run_qdrant.sh start

# 4. Start MCP Server (Force Restart)
echo "[4/7] Restarting MCP Server..."
./run_mcp_server.sh stop > /dev/null 2>&1
sleep 1
./run_mcp_server.sh start

# 5. Start TTS (Piper voices for Open WebUI)
echo "[5/7] Starting TTS (openedai-speech)..."
fuser -k 8001/tcp 2>/dev/null || true
sleep 1
./run_tts.sh start

# 6. Start Admin Portal
echo "[6/7] Starting Admin Portal..."
# Kill existing admin_server and clear port 5001
echo "  Stopping existing admin processes..."
pkill -f admin_server.py || true
fuser -k 5001/tcp > /dev/null 2>&1 || true
sleep 1
mkdir -p logs
nohup uv run python3 admin_server.py >> logs/do-rag.log 2>&1 &
echo "  Admin Portal running on port 5001"

# 7. Start Open WebUI (Already handles restart)
echo "[7/7] Starting Open WebUI..."
./run_webui.sh

echo "================================"
echo "All services verified up."
echo "Access Open WebUI at:"
echo "  http://localhost:3000"
echo "  http://$(hostname -I | awk '{print $1}'):3000"
echo "Access Admin Portal at:"
echo "  http://localhost:5001"
echo "================================"

# Optional: Import Local File Expert model preset (runs in background so run.sh doesn't block)
if [ -n "$OPENWEBUI_ADMIN_PASSWORD" ]; then
  (
    echo "Importing Local File Expert model preset (background)..."
    ADMIN_EMAIL="${OPENWEBUI_ADMIN_EMAIL:-admin@local}"
    for i in $(seq 1 30); do
      if curl -sf http://localhost:3000/health > /dev/null 2>&1; then
        TOKEN=$(curl -sf -X POST http://localhost:3000/api/v1/auths/signin \
          -H "Content-Type: application/json" \
          -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$OPENWEBUI_ADMIN_PASSWORD\"}" 2>/dev/null | \
          python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('token',d.get('access_token','')))" 2>/dev/null)
        if [ -n "$TOKEN" ]; then
          if curl -sf -X POST http://localhost:3000/api/v1/models/import \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            -d "@$SCRIPT_DIR/open_webui/local-file-expert.json" > /dev/null 2>&1; then
            echo "  Local File Expert model imported."
          else
            echo "  Import failed (model may already exist)."
          fi
        else
          echo "  Import skipped (login failed)."
        fi
        exit 0
      fi
      sleep 2
    done
    echo "  Import skipped (Open WebUI not ready within 60s)."
  ) &
fi
