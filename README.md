# Local Document RAG System

A privacy-preserving, local document search system using RAG (Retrieval-Augmented Generation) with [Ollama](https://ollama.com/) and [Qdrant](https://qdrant.tech/).

## Features

- **Local & Private**: Runs entirely on your machine — no data leaves your system.
- **Incremental Indexing**: Only processes new or modified files.
- **Robust Vector Search**: Uses Qdrant (Podman/Docker) for high-performance, crash-resilient vector storage.
- **MCP Server**: Exposes search tools via the [Model Context Protocol](https://modelcontextprotocol.io/) with dual transport support:
  - **Streamable HTTP** at `/mcp` — for Open WebUI and other HTTP-based clients
  - **SSE** at `/sse-transport/sse` — for IDEs (LM Studio, Claude Desktop, etc.)
- **Open WebUI Integration**: Chat with your documents through a web interface powered by Ollama.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────┐
│  Open WebUI  │────▶│   MCP Server     │────▶│  Qdrant   │
│  (Podman)    │     │  (server.py)     │     │ (Podman)  │
│  :3000       │     │  :8000           │     │ :6333     │
└──────────────┘     │                  │     └───────────┘
                     │  Streamable HTTP │           │
┌──────────────┐     │  + SSE           │     ┌───────────┐
│  IDE / CLI   │────▶│                  │────▶│  Ollama   │
│  (SSE)       │     └──────────────────┘     │  :11434   │
└──────────────┘                              └───────────┘
```

## Prerequisites

1. **Ollama**: Installed and available on `PATH`.
2. **Models**: Pull the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5-coder:7b-instruct-q5_K_M   # or your preferred LLM
   ```
3. **Python & uv**: This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.
4. **Podman or Docker**: Required for Qdrant and Open WebUI. Scripts prefer Podman when available; use `CONTAINER_CMD=docker ./run.sh` to force Docker.

## Installation

1. **Clone the repository** (or copy files).
2. **Install dependencies**:
   ```bash
   uv sync
   ```
3. **Configure Paths**:
   Set your document directories via the Admin UI at http://localhost:5001/config after starting services, or create `settings.json` in the project root:
   ```json
   {
     "DOCUMENT_DIRECTORIES": [
       "/home/yourname/Projects",
       "/home/yourname/Documents"
     ],
     "SUPPORTED_EXTENSIONS": [".md", ".txt", ".docx"]
   }
   ```

   > **Important**: Add all directories you want searchable. Use absolute paths. On WSL Linux, prefer `/home/...` paths for lower latency. Windows drives are mounted under `/mnt/c/`, `/mnt/d/`, etc.

## Quick Start

Start all services with one command:

```bash
./run.sh
```

On first run, you'll be prompted for an Open WebUI admin password (saved to `.env`). This creates the chat admin account and imports the Local File Expert model. Subsequent runs use the saved credentials.

Then open **http://localhost:3000** in your browser.

## Usage

### 1. Indexing Documents

Scan your configured directories and generate embeddings:

```bash
uv run index_docs.py
```

- **Incremental**: Subsequent runs only process changed files.
- **Force Reset**: `uv run index_docs.py --reset` (clears index and re-indexes).

### 2. Search (CLI)

Search your documents from the command line:

```bash
uv run search_docs.py "your search query"
```

### 3. MCP Server

The MCP server exposes two tools:

| Tool | Description |
|------|-------------|
| `search_documents` | Returns relevant document excerpts matching a query |
| `ask_documents` | Uses RAG to answer questions based on indexed content |

#### Running the Server

```bash
./run_mcp_server.sh start    # Start in background
./run_mcp_server.sh stop     # Stop
./run_mcp_server.sh status   # Check status
./run_mcp_server.sh logs     # Tail logs
```

The server starts with dual transport on port **8000**:

| Transport | Endpoint | Use Case |
|-----------|----------|----------|
| Streamable HTTP | `http://localhost:8000/mcp` | Open WebUI, HTTP clients |
| SSE | `http://localhost:8000/sse-transport/sse` | IDEs, LM Studio, Claude Desktop |

### 4. Open WebUI (Chat Interface)

#### Starting Open WebUI

```bash
./run_webui.sh
```

This runs Open WebUI as a Docker container at **http://localhost:3000**. Text-to-Speech (TTS) using Piper voices (openedai-speech, CPU-only) is pre-configured when `run_tts.sh` is started (included in `./run.sh`).

#### Connecting MCP Tools

**Pre-configured**: When you start Open WebUI via `./run_webui.sh`, the Local RAG Search MCP tool server is automatically configured. No manual import is needed for fresh installs.

If you have an existing Open WebUI volume from before this change, either:
- Remove the volume and restart for a clean install: `podman volume rm open-webui` (or `docker volume rm open-webui`), then `./run_webui.sh`
- Or manually import from [`open_webui/tool-server-do-rag.json`](open_webui/tool-server-do-rag.json): **Admin Panel → Settings → Tools → Import**

#### Using the Tools in Chat

1. Start a **New Chat**.
2. Click the **Integrations icon** (➕) near the message input.
3. Enable **Local RAG Search** (`do-rag`).
4. Ask a question about your documents, e.g.: *"Show me local files discussing motors."*

**Pre-load the Local File Expert model:** The model preset is imported automatically when you run `./run.sh` and provide the admin password during first-run setup. Credentials are stored in `.env`. To change them, edit `.env` or delete it and run `./run.sh` again to be re-prompted. Or import manually: **Workspace → Models → Import** and select `open_webui/local-file-expert.json`. Edit `base_model_id` in the JSON if your LLM differs (e.g. `qwen2.5-coder:7b`).

*Note: Some smaller models (e.g. Qwen Coder 7B) may not reliably invoke tools. Larger models (14B+) generally work better with MCP tools.*

### 5. IDE Integration (SSE)

#### LM Studio (v0.4.x+)

In the **Developer** tab, click **MCP.json** and paste:

```json
{
  "mcpServers": {
    "local-rag": {
      "type": "sse",
      "url": "http://localhost:8000/sse-transport/sse"
    }
  }
}
```

#### Claude Desktop / Other MCP Clients

For `stdio` mode (direct process spawning), use:

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "uv",
      "args": ["run", "server.py"],
      "cwd": "/path/to/do-rag"
    }
  }
}
```

## Service Management

| Script | Purpose |
|--------|---------|
| `run.sh` | Start all services (Ollama + Qdrant + MCP + Open WebUI) |
| `run_ollama.sh` | Manage Ollama (`start`/`stop`/`status`/`restart`) |
| `run_qdrant.sh` | Manage Qdrant (`start`/`stop`/`status`) |
| `run_mcp_server.sh` | Manage MCP server (`start`/`stop`/`status`/`logs`) |
| `run_tts.sh` | Manage TTS container (`start`/`stop`/`status`) — Piper voices (openedai-speech, CPU-only) for Open WebUI |
| `run_webui.sh` | Start/restart Open WebUI container (Podman/Docker) |

Logs are consolidated in `logs/do-rag.log` (Admin Portal, MCP server, indexer). View from the Admin UI when indexing reports errors, or run `tail -f logs/do-rag.log`. Ollama uses `ollama_server.log` in the project root.

## Configuration

**Credentials** are stored in `.env` (gitignored). See `.env.example` for the format. Run `./run.sh` to be prompted for missing values.

**Document paths and models** are in `settings.json` (or Admin UI at http://localhost:5001/config). Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `QDRANT_HOST` | `"localhost"` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant API port |
| `DOCUMENT_DIRECTORIES` | `[]` | Directories to index |
| `SUPPORTED_EXTENSIONS` | `.md`, `.txt`, `.docx` | File types to process |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `EMBEDDING_MODEL` | `nomic-rag` | Model for document embeddings |
| `LLM_MODEL` | `qwen2.5:14b` | Model for RAG answers |
| `INDEXING_BATCH_SIZE` | `10` | Chunks per Qdrant write (higher = faster) |

## Limitations

- **Supported file types**: The indexer supports **Markdown** (`.md`), **plain text** (`.txt`), and **Word documents** (`.docx`). Other formats like `.pdf`, `.xlsx`, and `.pptx` are not yet supported.
- **No real-time indexing**: Documents are indexed on-demand via `index_docs.py`. New or modified files won't be searchable until the indexer is re-run.

## Troubleshooting

### MCP server won't start
Check the logs: `./run_mcp_server.sh logs`

### Open WebUI shows "Failed to connect to MCP server"
- Ensure the MCP server is running: `./run_mcp_server.sh status`
- Verify the **Auth Type** is set to **None** (not Bearer) in the tool server config.
- Check that the URL uses `host.containers.internal` (Podman) or `host.docker.internal` (Docker) since Open WebUI runs in a container.

### Ollama connection errors
- Ensure Ollama is running: `./run_ollama.sh status`
- Verify the required models are pulled: `ollama list`

### Migrated from Windows to WSL Linux
If you moved the project from the Windows filesystem to the WSL Linux filesystem, update `settings.json` (or Admin UI config) to use Linux paths. Replace `/mnt/c/Users/...` with `/home/username/...` for documents now under the Linux filesystem. Prefer Linux paths for lower I/O latency.
