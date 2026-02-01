# Local Document RAG System

A privacy-preserving, local document search system using RAG (Retrieval-Augmented Generation) with [Ollama](https://ollama.com/) and [ChromaDB](https://www.trychroma.com/).

Features:
- **Local & Private**: Runs entirely on your machine.
- **Incremental Indexing**: Only processes new or modified files.
- **MCP Server**: Exposes search capabilities to AI agents (like LM Studio) via the [Model Context Protocol](https://modelcontextprotocol.io/).

## Prerequisites

1.  **Ollama**: Installed and running (`ollama serve`).
2.  **Embedding Model**: Pull the required model:
    ```bash
    ollama pull nomic-embed-text
    ```
3.  **Python & uv**: This project uses `uv` for dependency management.

## Installation

1.  **Clone the repository** (or copy files).
2.  **Install dependencies**:
    ```bash
    uv sync
    ```
3.  **Start the Ollama Service**:
    Use the service manager to start Ollama in the background.
    ```bash
    ./run_ollama.sh start
    ```
    - `start`: Starts service (background).
    - `stop`: Stops service.
    - `status`: Checks if running.
    - `restart`: Restarts service and truncates logs.
    
    *Logs are saved to `ollama_server.log`.*

4.  **Configure Paths**:
    Copy `config.example.py` to `config.py` and edit it to point to your document folders.
    ```bash
    cp config.example.py config.py
    # Edit config.py with your preferred editor
    ```

```python
DOCUMENT_DIRECTORIES = [
    "/mnt/c/Users/yourname/Documents",
    "/mnt/c/Users/yourname/Projects"
]
```

## Usage

### 1. Indexing Documents

Run the indexer to scan directories and generate embeddings.

```bash
uv run index_docs.py
```

- **Incremental**: Subsequent runs only process changed files.
- **Force Reset**: `uv run index_docs.py --reset` (clears index and restarts).

### 2. Search (CLI)

Search your documents from the command line:

```bash
uv run search_docs.py "your search query"
```

### 3. MCP Server (AI Agent Integration)

This system provides an MCP server that exposes a `search_documents` tool.

#### Running the Service
Use the background service script (logs saved to `mcp_server.log`):

```bash
./run_mcp_server.sh start
```
*Endpoint: `http://localhost:8000/sse`*

#### Connecting from LM Studio (v0.4.x)
1.  In the **Developer** tab, click the **MCP.json** button at the top (embedded editor).
2.  Paste this SSE configuration:

```json
{
  "mcpServers": {
    "local-rag": {
      "type": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

*Note: You can still run it manually with `uv run server.py`, but the background service is recommended for logs and stability.*
