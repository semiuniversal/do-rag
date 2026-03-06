# config.example.py
# Template for settings.json structure. The main config loads from settings.json.
# This file documents the expected keys. Copy to config.py only if using legacy Python config.

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()

# Document directories to index (use absolute paths)
DOCUMENT_DIRECTORIES = [
    str(PROJECT_ROOT / "test_docs"),
    # "/home/username/Projects",      # WSL Linux (recommended)
    # "/home/username/Documents",
    # "/mnt/c/Users/username/Documents",  # Windows via WSL mount
]

# File extensions to process
SUPPORTED_EXTENSIONS = ['.md', '.txt', '.docx', '.pdf', '.pptx', '.xlsx', '.html']

# LLM Backend: "ollama" or "lm_studio"
BACKEND = "ollama"

# Ollama settings (used when BACKEND = "ollama")
OLLAMA_BASE_URL = "http://localhost:11434"

# LM Studio settings (used when BACKEND = "lm_studio")
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# Model names (used by both backends — make sure these are loaded/pulled)
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:14b"

# Qdrant vector database
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "do_rag"

# Chunking configuration
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks

# Search configuration
TOP_K_RESULTS = 10  # number of results to return
