# config.example.py
# Copy this file to 'config.py' and update with your local paths.

import os
from pathlib import Path

# Document directories to index
DOCUMENT_DIRECTORIES = [
    # Windows paths (WSL)
    # "/mnt/c/Users/username/Documents",
    
    # macOS paths
    # "/Users/username/Documents",

    # Linux paths
    # "/home/username/documents",
]

# File extensions to process
SUPPORTED_EXTENSIONS = ['.md', '.txt', '.docx']

# LLM Backend: "ollama" or "lm_studio"
BACKEND = "ollama"

# Ollama settings (used when BACKEND = "ollama")
OLLAMA_BASE_URL = "http://localhost:11434"

# LM Studio settings (used when BACKEND = "lm_studio")
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# Model names (used by both backends — make sure these are loaded/pulled)
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:7b-instruct-q5_K_M"

# ChromaDB configuration
PROJECT_ROOT = Path(__file__).parent.absolute()
CHROMA_PERSIST_DIRECTORY = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "documents"

# Chunking configuration
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks

# Search configuration
TOP_K_RESULTS = 10  # number of results to return
