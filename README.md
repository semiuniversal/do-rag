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
3.  **Setup the Custom Model**:
    This project requires a custom Ollama model configuration to handle large batches.
    ```bash
    bash setup_model.sh
    ```
    *This creates a `nomic-rag` model in Ollama with an 8192 token window.*

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
