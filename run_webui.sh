# Get the WSL host IP (eth0)
HOST_IP=$(ip -4 addr show eth0 | awk '/inet/ {print $2}' | cut -d/ -f1)

docker rm -f open-webui 2>/dev/null || true
docker run -d -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e WEBUI_SECRET_KEY=do-rag-secret-key-2026-long-enough-now \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --add-host=host.docker.internal:$HOST_IP \
  ghcr.io/open-webui/open-webui:main