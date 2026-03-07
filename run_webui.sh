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

# Pre-configure the do-rag MCP tool server (Local RAG Search) so it appears without manual import
MCP_URL="http://${HOST_ALIAS}:8000/mcp"
TOOL_SERVER_JSON="[{\"type\":\"mcp\",\"url\":\"${MCP_URL}\",\"spec_type\":\"url\",\"spec\":\"\",\"path\":\"openapi.json\",\"auth_type\":\"none\",\"key\":\"\",\"config\":{\"enable\":true},\"info\":{\"id\":\"do-rag\",\"name\":\"Local RAG Search\",\"description\":\"Search and ask questions about your indexed documents.\"}}]"

# Pre-configure TTS (openedai-speech Piper on host port 8001)
TTS_URL="http://${HOST_ALIAS}:8001/v1"

# Admin account for first-run (enables model import without API key)
WEBUI_ADMIN_EMAIL="${OPENWEBUI_ADMIN_EMAIL:-admin@local}"
WEBUI_ADMIN_PASSWORD="${OPENWEBUI_ADMIN_PASSWORD:-}"

$CONTAINER_CMD rm -f open-webui 2>/dev/null || true
$CONTAINER_CMD run -d -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://${HOST_ALIAS}:11434 \
  -e WEBUI_SECRET_KEY=do-rag-secret-key-2026-long-enough-now \
  -e WEBUI_ADMIN_EMAIL="${WEBUI_ADMIN_EMAIL}" \
  -e WEBUI_ADMIN_PASSWORD="${WEBUI_ADMIN_PASSWORD}" \
  -e TOOL_SERVER_CONNECTIONS="${TOOL_SERVER_JSON}" \
  -e USER_PERMISSIONS_FEATURES_DIRECT_TOOL_SERVERS=true \
  -e AUDIO_TTS_ENGINE=openai \
  -e AUDIO_TTS_OPENAI_API_BASE_URL="${TTS_URL}" \
  -e AUDIO_TTS_OPENAI_API_KEY=your_api_key_here \
  -e AUDIO_TTS_MODEL=tts-1 \
  -e AUDIO_TTS_VOICE=nova \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --add-host=${HOST_ALIAS}:${HOST_IP} \
  ghcr.io/open-webui/open-webui:main