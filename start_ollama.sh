#!/bin/bash
# Start Ollama server and capture logs to both console and file
echo "Starting Ollama server..."
echo "Logs will be saved to: ollama_server.log"
echo "Press Ctrl+C to stop."

# Ensure log file exists (touch)
touch ollama_server.log

# Run ollama serve, redirect stderr (2) to stdout (1), then tee to file
ollama serve 2>&1 | tee -a ollama_server.log
