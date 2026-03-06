#!/bin/bash
# run_webui.sh — Start Open WebUI container (Podman or Docker)

# Prefer Podman, fallback to Docker
CONTAINER_CMD="${CONTAINER_CMD:-$(command -v podman 2>/dev/null || command -v docker 2>/dev/null || echo docker)}"

# Podman uses host.containers.internal; Docker uses host.docker.internal
HOST_ALIAS="${HOST_ALIAS:-$( ($CONTAINER_CMD --version 2>/dev/null | grep -qi podman) && echo host.containers.internal || echo host.docker.internal)}"

# Get the WSL host IP (eth0) for container-to-host routing
HOST_IP=$(ip -4 addr show eth0 2>/dev/null | awk '/inet/ {print $2}' | cut -d/ -f1)
if [ -z "$HOST_IP" ]; then
  HOST_IP=host-gateway
fi

$CONTAINER_CMD rm -f open-webui 2>/dev/null || true
$CONTAINER_CMD run -d -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://${HOST_ALIAS}:11434 \
  -e WEBUI_SECRET_KEY=do-rag-secret-key-2026-long-enough-now \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --add-host=${HOST_ALIAS}:${HOST_IP} \
  ghcr.io/open-webui/open-webui:main