# Local Document RAG System

A privacy-preserving, local document search system using RAG (Retrieval-Augmented Generation) with [Ollama](https://ollama.com/) and [ChromaDB](https://www.trychroma.com/).

## Features

- **Local & Private**: Runs entirely on your machine — no data leaves your system.
- **Incremental Indexing**: Only processes new or modified files.
- **MCP Server**: Exposes search tools via the [Model Context Protocol](https://modelcontextprotocol.io/) with dual transport support:
  - **Streamable HTTP** at `/mcp` — for Open WebUI and other HTTP-based clients
  - **SSE** at `/sse-transport/sse` — for IDEs (LM Studio, Claude Desktop, etc.)
- **Open WebUI Integration**: Chat with your documents through a web interface powered by Ollama.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────┐
│  Open WebUI  │────▶│   MCP Server     │────▶│ ChromaDB  │
│  (Docker)    │     │  (server.py)     │     │           │
│  :3000       │     │  :8000           │     └───────────┘
└──────────────┘     │                  │           │
                     │  Streamable HTTP │     ┌───────────┐
┌──────────────┐     │  + SSE           │────▶│  Ollama   │
│  IDE / CLI   │────▶│                  │     │  :11434   │
│  (SSE)       │     └──────────────────┘     └───────────┘
└──────────────┘
```

## Prerequisites

1. **Ollama**: Installed and available on `PATH`.
2. **Models**: Pull the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5-coder:7b-instruct-q5_K_M   # or your preferred LLM
   ```
3. **Python & uv**: This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.
4. **Docker**: Required for Open WebUI.

## Installation

1. **Clone the repository** (or copy files).
2. **Install dependencies**:
   ```bash
   uv sync
   ```
3. **Configure Paths**:
   Copy `config.example.py` to `config.py` and edit it:
   ```bash
   cp config.example.py config.py
   ```
   Set your document directories:
   ```python
   DOCUMENT_DIRECTORIES = [
       "/mnt/c/Users/yourname/Documents",
       "/mnt/c/Users/yourname/Projects"
   ]
   ```

## Quick Start

Start all services (Ollama, MCP Server, Open WebUI) with one command:

```bash
./run.sh
```

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

This runs Open WebUI as a Docker container at **http://localhost:3000**.

#### Connecting MCP Tools

**Quick Setup**: Import the pre-configured tool server from [`open_webui/tool-server-do-rag.json`](open_webui/tool-server-do-rag.json):

1. Go to **Admin Panel → Settings → Tools**.
2. Click the **Import** button and select the JSON file.
3. Click **Verify Connection** — it should show a green checkmark.
4. Click **Save**.

<details>
<summary><strong>Manual Setup</strong> (if not importing)</summary>

1. Go to **Admin Panel → Settings → Tools**.
2. Add a new tool server with these settings:

   | Setting | Value |
   |---------|-------|
   | **Type** | MCP (Streamable HTTP) |
   | **URL** | `http://host.docker.internal:8000/mcp` |
   | **Auth Type** | None |
   | **ID** | `do-rag` |
   | **Name** | `Local RAG Search` |

3. Click **Verify Connection** — it should show a green checkmark.
4. Click **Save**.

</details>

#### Using the Tools in Chat

1. Start a **New Chat**.
2. Click the **Integrations icon** (plug icon) near the message input.
3. Enable the MCP tools (`do-rag`).
4. Ask a question about your documents, e.g.: *"Show me local files discussing motors."*

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
| `run.sh` | Start all services (Ollama + MCP + Open WebUI) |
| `run_ollama.sh` | Manage Ollama (`start`/`stop`/`status`/`restart`) |
| `run_mcp_server.sh` | Manage MCP server (`start`/`stop`/`status`/`logs`) |
| `run_webui.sh` | Start/restart Open WebUI Docker container |

Logs are saved to `*.log` files in the project root:
- `ollama_server.log` — Ollama output
- `mcp_server.log` — MCP server output

## Configuration

Edit `config.py` (copied from `config.example.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `DOCUMENT_DIRECTORIES` | `[]` | Directories to index |
| `SUPPORTED_EXTENSIONS` | `.md`, `.txt`, `.docx` | File types to process |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model for document embeddings |
| `LLM_MODEL` | `qwen2.5-coder:7b-instruct-q5_K_M` | Model for RAG answers |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `10` | Number of search results |

## Troubleshooting

### MCP server won't start
Check the logs: `./run_mcp_server.sh logs`

### Open WebUI shows "Failed to connect to MCP server"
- Ensure the MCP server is running: `./run_mcp_server.sh status`
- Verify the **Auth Type** is set to **None** (not Bearer) in the tool server config.
- Check that the URL uses `host.docker.internal` (not `localhost`) since Open WebUI runs in Docker.

### Ollama connection errors
- Ensure Ollama is running: `./run_ollama.sh status`
- Verify the required models are pulled: `ollama list`
