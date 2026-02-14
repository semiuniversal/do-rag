#!/bin/bash
# run_qdrant.sh — Manage the Qdrant vector database container

CONTAINER_NAME="qdrant"
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
IMAGE="qdrant/qdrant:latest"

case "${1:-start}" in
  start)
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      echo "Qdrant is already running."
      exit 0
    fi

    # Remove stopped container if it exists
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo "Starting Qdrant..."
    docker run -d \
      --name "$CONTAINER_NAME" \
      -p ${QDRANT_PORT}:6333 \
      -p ${QDRANT_GRPC_PORT}:6334 \
      -v qdrant_data:/qdrant/storage \
      --restart unless-stopped \
      "$IMAGE"

    # Wait for health
    echo -n "Waiting for Qdrant to be ready..."
    for i in $(seq 1 30); do
      if curl -sf http://localhost:${QDRANT_PORT}/healthz > /dev/null 2>&1; then
        echo " ready!"
        exit 0
      fi
      echo -n "."
      sleep 1
    done
    echo " timeout! Check: docker logs $CONTAINER_NAME"
    exit 1
    ;;

  stop)
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      docker stop "$CONTAINER_NAME" > /dev/null
      echo "Qdrant stopped."
    else
      echo "Qdrant is not running."
    fi
    ;;

  status)
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      echo "Qdrant is running on port ${QDRANT_PORT}"
      curl -sf http://localhost:${QDRANT_PORT}/healthz && echo " (healthy)" || echo " (unhealthy)"
    else
      echo "Qdrant is not running."
    fi
    ;;

  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
