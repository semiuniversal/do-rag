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

1.  Clone this repository.
2.  Install dependencies:
    ```bash
    uv sync
    ```

## Configuration

1.  Copy `config.example.py` to `config.py`:
    ```bash
    cp config.example.py config.py
    ```
2.  Edit `config.py` to set your document paths:

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

#### Running Locally (WSL/Linux)
```bash
uv run server.py
```

#### Connecting from LM Studio (Windows) to WSL

If you are running LM Studio on Windows and this project in WSL, configure your MCP server in LM Studio with the following settings to bridge the connection:

**Command**: `wsl.exe`
**Args**:
- `-e`
- `bash`
- `-c`
- `cd /absolute/path/to/do-rag && uv run server.py`

*Replace `/absolute/path/to/do-rag` with the full path to this directory in WSL.*

**Example JSON Config:**
```json
{
  "local-rag": {
    "command": "wsl.exe",
    "args": [
      "-e",
      "bash",
      "-c",
      "cd /mnt/c/Users/wtrem/Projects/do-rag && uv run server.py"
    ]
  }
}
```
