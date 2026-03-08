#!/bin/bash
# stop.sh — Halt all do-rag stack services and abort any running indexing

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Prefer Podman, fallback to Docker (for container stops)
CONTAINER_CMD="${CONTAINER_CMD:-$(command -v podman 2>/dev/null || command -v docker 2>/dev/null || echo docker)}"

echo "=== Stopping do-rag Stack ==="

# 1. Abort indexing (biggest time sink) — cancel via API if admin is reachable
echo "[1/6] Aborting indexing (if running)..."
if curl -sf -X POST http://localhost:5001/api/indexer/stop > /dev/null 2>&1; then
  echo "  Indexing cancelled."
  sleep 2  # Allow indexer thread to exit gracefully
else
  echo "  Admin not reachable (may already be stopped)."
fi

# 2. Stop Admin Portal (kills indexer thread if still running)
echo "[2/6] Stopping Admin Portal..."
pkill -f admin_server.py 2>/dev/null || true
fuser -k 5001/tcp 2>/dev/null || true
sleep 1

# 3. Stop Open WebUI container
echo "[3/6] Stopping Open WebUI..."
$CONTAINER_CMD stop open-webui 2>/dev/null || true
$CONTAINER_CMD rm -f open-webui 2>/dev/null || true

# 4. Stop MCP Server
echo "[4/6] Stopping MCP Server..."
./run_mcp_server.sh stop > /dev/null 2>&1

# 5. Stop TTS container and free port 8001 (rootlessport may hold it briefly)
echo "[5/6] Stopping TTS..."
./run_tts.sh stop > /dev/null 2>&1
fuser -k 8001/tcp 2>/dev/null || true
sleep 1

# 6. Stop Qdrant and Ollama
echo "[6/6] Stopping Qdrant and Ollama..."
./run_qdrant.sh stop > /dev/null 2>&1
./run_ollama.sh stop > /dev/null 2>&1

echo "================================"
echo "All services stopped."
