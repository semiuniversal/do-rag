import os
from pathlib import Path

# Document directories to index
# Update these paths to match your actual environment
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
PROJECT_ROOT = Path(__file__).parent.absolute()
CHROMA_PERSIST_DIRECTORY = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "documents"

# Chunking configuration
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks

# Search configuration
TOP_K_RESULTS = 10  # number of results to return
