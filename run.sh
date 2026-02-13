#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Starting do-rag Services ==="

# 1. Start Ollama (Force Restart to ensure port binding)
echo "[1/3] Restarting Ollama..."
./run_ollama.sh stop > /dev/null 2>&1
sleep 1
./run_ollama.sh start

# 2. Start MCP Server (Force Restart)
echo "[2/3] Restarting MCP Server..."
./run_mcp_server.sh stop > /dev/null 2>&1
sleep 1
./run_mcp_server.sh start

# 3. Start Open WebUI (Already handles restart)
echo "[3/3] Starting Open WebUI..."
./run_webui.sh

echo "================================"
echo "All services verified up."
echo "Access Open WebUI at:"
echo "  http://localhost:3000"
echo "  http://$(hostname -I | awk '{print $1}'):3000"
echo "================================"
