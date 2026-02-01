#!/bin/bash

# Configuration
MODEL_NAME="nomic-rag"
LOG_FILE="ollama_server.log"

echo "=== Ollama RAG Service Manager ==="

# 1. Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: 'ollama' command not found. Please install Ollama first."
    exit 1
fi

# 2. Start Ollama Server in Background
echo "Starting Ollama server..."
echo "Logs are being saved to: $LOG_FILE"
touch "$LOG_FILE"
ollama serve > >(tee -a "$LOG_FILE") 2>&1 &
SERVER_PID=$!

# Ensure we kill the server when this script exits
trap "kill $SERVER_PID" EXIT

# 3. Wait for Server to be Ready
echo "Waiting for Ollama API to be ready..."
RETRIES=0
MAX_RETRIES=30
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
    RETRIES=$((RETRIES+1))
    if [ $RETRIES -ge $MAX_RETRIES ]; then
        echo "Error: Timed out waiting for Ollama server."
        exit 1
    fi
done
echo "Ollama server is up."

# 4. Check Base Model Dependency
BASE_MODEL="nomic-embed-text"
if ollama list | grep -q "$BASE_MODEL"; then
    echo "Base model '$BASE_MODEL' found."
else
    echo "Base model '$BASE_MODEL' NOT found. Pulling it now..."
    ollama pull "$BASE_MODEL"
fi

# 5. Check/Create Custom Model
if ollama list | grep -q "$MODEL_NAME"; then
    echo "Model '$MODEL_NAME' already exists. Good."
else
    echo "Model '$MODEL_NAME' not found. Creating from Modelfile..."
    if [ -f Modelfile ]; then
        ollama create "$MODEL_NAME" -f Modelfile
        echo "Model created successfully."
    else
        echo "Error: Modelfile not found in current directory!"
        exit 1
    fi
fi

# 5. Monitor Logs
echo "=== Service Running ==="
echo "Press Ctrl+C to stop the server."
echo "Tailing logs:"
tail -f "$LOG_FILE" --pid=$SERVER_PID
