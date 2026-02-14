#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Starting do-rag Services ==="

# 1. Start Ollama (Force Restart to ensure port binding)
echo "[1/5] Restarting Ollama..."
./run_ollama.sh stop > /dev/null 2>&1
sleep 1
./run_ollama.sh start

# 2. Pull required models if missing
echo "[2/5] Checking models..."
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

# 3. Start Qdrant (Docker)
echo "[3/5] Starting Qdrant..."
./run_qdrant.sh start

# 4. Start MCP Server (Force Restart)
echo "[4/5] Restarting MCP Server..."
./run_mcp_server.sh stop > /dev/null 2>&1
sleep 1
./run_mcp_server.sh start

# 5. Start Open WebUI (Already handles restart)
echo "[5/5] Starting Open WebUI..."
./run_webui.sh

echo "================================"
echo "All services verified up."
echo "Access Open WebUI at:"
echo "  http://localhost:3000"
echo "  http://$(hostname -I | awk '{print $1}'):3000"
echo "================================"
