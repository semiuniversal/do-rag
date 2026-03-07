#!/bin/bash
# run_tts.sh — Manage the openedai-speech container (Piper TTS, CPU-only, no GPU)

# Prefer Podman, fallback to Docker
CONTAINER_CMD="${CONTAINER_CMD:-$(command -v podman 2>/dev/null || command -v docker 2>/dev/null || echo docker)}"

CONTAINER_NAME="openedai-speech"
TTS_PORT=8001
IMAGE="ghcr.io/matatonic/openedai-speech-min:latest"

case "${1:-start}" in
  start)
    if $CONTAINER_CMD ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      echo "TTS (openedai-speech) is already running."
      exit 0
    fi

    # Remove stopped container if it exists
    $CONTAINER_CMD rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo "Starting TTS ($CONTAINER_CMD) — Piper voices, CPU-only..."
    $CONTAINER_CMD run -d \
      --name "$CONTAINER_NAME" \
      -p ${TTS_PORT}:8000 \
      -v openedai-speech-voices:/app/voices \
      -v openedai-speech-config:/app/config \
      --restart unless-stopped \
      "$IMAGE"

    # Wait for readiness (Piper loads models on first request; /v1/ models endpoint may exist)
    echo -n "Waiting for TTS to be ready..."
    for i in $(seq 1 45); do
      if curl -s -o /dev/null -w "%{http_code}" http://localhost:${TTS_PORT}/ 2>/dev/null | grep -qE '^[2-5][0-9]{2}$'; then
        echo " ready!"
        exit 0
      fi
      echo -n "."
      sleep 1
    done
    echo " timeout! Check: $CONTAINER_CMD logs $CONTAINER_NAME"
    exit 1
    ;;

  stop)
    if $CONTAINER_CMD ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      $CONTAINER_CMD stop "$CONTAINER_NAME" > /dev/null
      echo "TTS stopped."
    else
      echo "TTS is not running."
    fi
    ;;

  status)
    if $CONTAINER_CMD ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      echo "TTS is running on port ${TTS_PORT}"
      curl -s -o /dev/null http://localhost:${TTS_PORT}/ 2>/dev/null && echo " (healthy)" || echo " (unhealthy)"
    else
      echo "TTS is not running."
    fi
    ;;

  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
