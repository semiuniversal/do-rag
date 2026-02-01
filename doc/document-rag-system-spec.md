# Local Document RAG System - Technical Specification

## Project Overview

Build a privacy-preserving, local document search system using Retrieval-Augmented Generation (RAG) to enable semantic search across 2+ years of markdown files, AI conversation transcripts, and project documentation.

### Core Goals

1. **Semantic search** - Find documents by meaning/concept, not just filename matching
2. **Privacy-first** - All processing happens locally, no cloud services
3. **Crutch not coaching** - Simple to use, minimal maintenance, no complex workflows
4. **One-time indexing** - Batch process once, query repeatedly without re-indexing

### User Context

- **Hardware**: Windows 11 laptop, WSL2 Ubuntu available
- **Existing tools**: Ollama running in WSL with `nomic-embed-text` model already pulled
- **Data location**: ~2 years of documents across multiple directories
- **Use case**: Finding conceptually-related content (e.g., "second brain concepts", "Docker authentication issues")

## System Architecture

```
User Query
    ↓
search_docs.py (Python)
    ↓
Query → Embedding (via Ollama)
    ↓
Vector Similarity Search (ChromaDB)
    ↓
Retrieve Top-K Chunks
    ↓
Return Results (filepath + excerpts)
```

### Components

1. **Ollama** (already installed/running in WSL)
   - Provides embedding generation via `nomic-embed-text` model
   - Running at `http://localhost:11434`

2. **ChromaDB** (to install)
   - Local vector database
   - Persistent storage at `./chroma_db/`
   - Stores document chunks + embeddings

3. **Python Scripts** (to create)
   - `index_docs.py` - One-time indexing of all documents
   - `search_docs.py` - Query interface for daily use

## Technical Requirements

### Dependencies

```bash
pip install langchain langchain-community chromadb ollama python-dotenv
```

### System Requirements

- Python 3.8+
- Ollama running with `nomic-embed-text` model
- ~2GB disk space for ChromaDB (depends on document corpus size)
- WSL2 or native Linux/Windows Python environment

## File Structure

```
~/document-rag/
├── index_docs.py          # Indexing script
├── search_docs.py         # Search script
├── config.py              # Configuration (directories, settings)
├── requirements.txt       # Python dependencies
├── chroma_db/            # Vector database (created on first run)
└── README.md             # Usage instructions
```

## Configuration Specification

### config.py

```python
# Document directories to index
DOCUMENT_DIRECTORIES = [
    "/mnt/c/Users/wtrem/Downloads",
    "/mnt/c/Users/wtrem/OneDrive/Desktop",
    "/mnt/c/Users/wtrem/Projects"
]

# File extensions to process
SUPPORTED_EXTENSIONS = ['.md', '.txt', '.docx']

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"

# ChromaDB configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "documents"

# Chunking configuration
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks

# Search configuration
TOP_K_RESULTS = 10  # number of results to return
```

## Component Specifications

### 1. index_docs.py

**Purpose**: One-time batch processing to index all documents

**Workflow**:
1. Scan all configured directories recursively
2. Filter for supported file extensions
3. Read file contents (handle .docx with python-docx)
4. Split documents into chunks with overlap
5. Generate embeddings via Ollama for each chunk
6. Store chunks + embeddings + metadata in ChromaDB
7. Report progress (files processed, chunks created, time elapsed)

**Key Features**:
- Skip hidden files/directories (starting with `.`)
- Handle encoding errors gracefully (try UTF-8, fallback to latin-1)
- Store metadata: filepath, chunk_index, file_modified_date
- Progress indicators every 10 files
- Error logging (continue on file errors, don't crash)
- Final summary: total files, total chunks, time taken

**Error Handling**:
- Log files that fail to process
- Continue processing on individual file errors
- Validate Ollama connectivity before starting
- Create ChromaDB directory if doesn't exist

**Expected Runtime**: 1-3 hours for ~1000 files

### 2. search_docs.py

**Purpose**: Interactive search interface for querying indexed documents

**Workflow**:
1. Accept query from command line argument or interactive prompt
2. Generate embedding for query via Ollama
3. Perform vector similarity search in ChromaDB
4. Return top-K results with:
   - Filepath (relative to search directories)
   - Chunk content (excerpt showing context)
   - Relevance score
   - File modified date

**Key Features**:
- Command-line usage: `python search_docs.py "second brain concepts"`
- Interactive mode if no query provided
- Pretty-printed results (filepath, score, excerpt)
- Option to show full file path vs. relative path
- Results sorted by relevance score (descending)

**Output Format**:
```
Query: "second brain concepts"

Result 1 (score: 0.87)
File: /mnt/c/Users/wtrem/Downloads/ai_conversations_2024-03.md
Modified: 2024-03-15
---
...excerpt showing matched content with context...
---

Result 2 (score: 0.82)
...
```

### 3. requirements.txt

```
langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
ollama>=0.1.0
python-dotenv>=1.0.0
python-docx>=1.1.0
```

## Implementation Details

### Document Chunking Strategy

Use `RecursiveCharacterTextSplitter` with:
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Separators: `["\n\n", "\n", ". ", " ", ""]`

**Rationale**: 
- 1000 chars ≈ 250 tokens, fits comfortably in embedding context
- 200 char overlap ensures context isn't lost at boundaries
- Recursive splitting preserves paragraph/sentence structure

### Embedding Generation

Use Ollama's `nomic-embed-text` model:
- Dimension: 768
- Max tokens: 2048
- Fast inference (~10ms per chunk)

**Connection**:
```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text"
)
```

### ChromaDB Configuration

```python
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

**Metadata Schema**:
```python
{
    "source": "/full/path/to/file.md",
    "chunk_index": 0,
    "file_modified": "2024-03-15T10:30:00",
    "file_size": 12345
}
```

### Error Handling Patterns

**File Reading**:
```python
try:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
except UnicodeDecodeError:
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()
except Exception as e:
    print(f"Error reading {filepath}: {e}")
    continue
```

**Ollama Connection**:
```python
try:
    # Test connection
    embeddings.embed_query("test")
except Exception as e:
    print(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    print(f"Error: {e}")
    print("Please ensure Ollama is running: ollama serve")
    sys.exit(1)
```

## Usage Instructions

### Initial Setup

```bash
# Create project directory
mkdir ~/document-rag
cd ~/document-rag

# Install dependencies
pip install -r requirements.txt

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Indexing (One-time)

```bash
# Run indexing (will take 1-3 hours)
python index_docs.py

# Expected output:
# Scanning directories...
# Found 1,247 files to process
# Processing files... [████████████] 100%
# Created 15,823 chunks
# Indexing complete in 2h 15m
```

### Searching (Daily use)

```bash
# Command-line query
python search_docs.py "second brain concepts"

# Interactive mode
python search_docs.py
> Enter query: Docker authentication issues
> [results shown]
> Enter query: retirement planning diversification
> [results shown]
> Enter query: quit
```

## Validation Criteria

### Success Metrics

1. **Indexing completes without crashing** on full document corpus
2. **Search returns semantically relevant results** for test queries:
   - "second brain concepts" → finds AI-augmented storage discussions
   - "Docker authentication issues" → finds AIDA debugging sessions
   - "retirement planning" → finds financial analysis documents
3. **Query latency < 2 seconds** for search operations
4. **No manual maintenance required** after initial indexing

### Test Queries

Include these in README for validation:

```bash
# Should find AI/knowledge management discussions
python search_docs.py "second brain AI augmented memory"

# Should find technical troubleshooting
python search_docs.py "Docker bearer token authentication"

# Should find financial planning content
python search_docs.py "retirement portfolio diversification"

# Should find artistic practice documentation
python search_docs.py "esthesiology perception making"
```

## Future Enhancements (Out of Scope for POC)

1. **Incremental indexing** - Watch directories for new/modified files
2. **Web interface** - Flask/FastAPI frontend for easier querying
3. **Multi-modal search** - Include images, PDFs with OCR
4. **Result ranking** - Combine vector similarity with keyword matching
5. **Export/sharing** - Generate markdown reports from search results

## Non-Requirements

The following are explicitly OUT OF SCOPE:

- ❌ Complex query syntax or boolean operators
- ❌ Real-time indexing or file watching
- ❌ Multi-user support or access control
- ❌ Cloud sync or backup
- ❌ LLM-generated summaries (just return chunks)
- ❌ GUI/web interface (CLI only for POC)

## Expected Deliverables

1. **index_docs.py** - ~100 lines, well-commented
2. **search_docs.py** - ~50 lines, well-commented  
3. **config.py** - ~30 lines with configuration constants
4. **requirements.txt** - Python dependencies
5. **README.md** - Setup and usage instructions

## Testing Checklist

- [ ] Ollama connectivity test passes
- [ ] Indexing processes all file types (.md, .txt, .docx)
- [ ] ChromaDB persists between runs
- [ ] Search returns results for known queries
- [ ] Error handling works (bad files, encoding issues)
- [ ] Progress indicators show during indexing
- [ ] Results include filepath, score, and excerpt

## Notes for Implementation

- Use `pathlib.Path` for cross-platform file handling
- WSL paths: `/mnt/c/Users/...` maps to `C:\Users\...`
- Prefer `argparse` for CLI argument parsing
- Use `tqdm` for progress bars (optional but nice)
- Include docstrings for all functions
- Keep code simple and readable (prefer clarity over cleverness)

---

**End of Specification**
