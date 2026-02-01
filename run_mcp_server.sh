#!/bin/bash

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVER_CMD="uv run server.py --transport sse --host 0.0.0.0"
PID_FILE="${SCRIPT_DIR}/mcp_server.pid"
LOG_FILE="${SCRIPT_DIR}/mcp_server.log"

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "MCP Server is already running with PID $PID."
                exit 1
            else
                echo "Found stale PID file. Removing."
                rm "$PID_FILE"
            fi
        fi

        echo "Starting MCP Server (SSE Mode)..."
        # Truncate log file on new start for cleanliness
        echo "--- Session Start: $(date) ---" > "$LOG_FILE"
        
        # Run in background
        nohup $SERVER_CMD >> "$LOG_FILE" 2>&1 &
        
        # Save PID
        echo $! > "$PID_FILE"
        echo "MCP Server started with PID $(cat "$PID_FILE"). Logs: $LOG_FILE"
        echo "Endpoint: http://localhost:8000/sse"
        ;;
    
    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "Stopping MCP Server (PID $PID)..."
                kill "$PID"
                rm "$PID_FILE"
                echo "MCP Server stopped."
            else
                echo "Process $PID not running. Cleaning up PID file."
                rm "$PID_FILE"
            fi
        else
            echo "No PID file found. Is the server running?"
        fi
        ;;
    
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "MCP Server is running (PID $PID)."
                echo "Last 5 log lines:"
                tail -n 5 "$LOG_FILE"
            else
                echo "MCP Server is NOT running (Stale PID file found)."
            fi
        else
            echo "MCP Server is NOT running."
        fi
        ;;
        
    logs)
        tail -f "$LOG_FILE"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status|logs}"
        exit 1
        ;;
esac
