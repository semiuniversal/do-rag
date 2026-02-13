#!/bin/bash

# Configuration
MODEL_NAME="nomic-rag"
BASE_MODEL="nomic-embed-text"
LOG_FILE="ollama_server.log"
PID_FILE="ollama.pid"

# Suppress Warnings
export OLLAMA_LOG_LEVEL=ERROR
export OLLAMA_DEBUG=0
export OLLAMA_HOST=0.0.0.0
export GIN_MODE=release

check_ready() {
    RETRIES=0
    MAX_RETRIES=30
    echo -n "Waiting for API..."
    while ! curl -s http://localhost:11434/api/tags > /dev/null; do
        sleep 1
        echo -n "."
        RETRIES=$((RETRIES+1))
        if [ $RETRIES -ge $MAX_RETRIES ]; then
            echo " Timeout!"
            return 1
        fi
    done
    echo " Ready."
    return 0
}

check_models() {
    # Check Base Embedding Model
    if ollama list | grep -q "$BASE_MODEL"; then
        echo "Base model '$BASE_MODEL' found."
    else
        echo "Base model '$BASE_MODEL' NOT found. Pulling..."
        ollama pull "$BASE_MODEL"
    fi

    # Check LLM Model
    # Extract model name from config or hardcode for script simplicity if needed, 
    # but here we use a variable corresponding to config
    LLM_MODEL="qwen2.5-coder:7b-instruct-q5_K_M"
    
    if ollama list | grep -q "$LLM_MODEL"; then
         echo "LLM model '$LLM_MODEL' found."
    else
         echo "LLM model '$LLM_MODEL' NOT found. Pulling (this may take a while)..."
         ollama pull "$LLM_MODEL"
    fi

    # Check Custom Embedding Model (if still using custom modelfile for embeddings)
    if ollama list | grep -q "$MODEL_NAME"; then
        echo "Model '$MODEL_NAME' ready."
    else
        echo "Model '$MODEL_NAME' missing. Creating..."
        if [ -f Modelfile ]; then
            ollama create "$MODEL_NAME" -f Modelfile
        else
            echo "Error: Modelfile not found!"
            return 1
        fi
    fi
}

start_server() {
    if [ -f "$PID_FILE" ]; then
        if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "Ollama is already running (PID $(cat $PID_FILE))."
            return
        else
            echo "Removing stale PID file."
            rm "$PID_FILE"
        fi
    fi

    echo "Starting Ollama server..."
    # Clear log on start
    > "$LOG_FILE"
    
    # Start with nohup correctly detaching stdin/stdout/stderr
    nohup ollama serve > "$LOG_FILE" 2>&1 < /dev/null &
    PID=$!
    echo $PID > "$PID_FILE"
    
    echo "Server started with PID $PID. Logs at $LOG_FILE"
    
    if check_ready; then
        check_models
        echo "=== Service Running ==="
        echo "Ollama is running in the background."
        echo "Use './run_ollama.sh stop' to stop it."
        echo "Use 'tail -f $LOG_FILE' to watch logs."
    else
        echo "Failed to start server."
        cat "$LOG_FILE"
        kill $PID 2>/dev/null
        rm "$PID_FILE"
        exit 1
    fi
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping Ollama (PID $PID)..."
        kill $PID 2>/dev/null
        rm "$PID_FILE"
        echo "Stopped."
    else
        echo "No PID file found. Is it running?"
    fi
}

status_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            echo "Ollama is RUNNING (PID $PID)"
        else
            echo "Ollama is NOT RUNNING (Stale PID file)"
        fi
    else
        echo "Ollama is NOT RUNNING"
    fi
}

# Main CLI dispatch
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    status)
        status_server
        ;;
    *)
        # Default behavior: Start if not running, or just show status
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "Ollama is already running."
            status_server
        else
            start_server
        fi
        ;;
esac
