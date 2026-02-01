import os
import sys
print("Loading RAG system dependencies...", file=sys.stderr)
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
import logging

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        for item in iterable:
            yield item

from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
try:
    from langchain_community.document_loaders import Docx2txtLoader
except ImportError:
    Docx2txtLoader = None

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

# Import local config
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indexing.log"),
        logging.StreamHandler()
    ]
)

STATE_FILE = "indexing_state.json"

def load_state() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load state file: {e}")
    return {}

def save_state(state: Dict[str, Dict[str, Any]]):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def get_files_to_index(directories: List[str], extensions: List[str]) -> List[Path]:
    """Recursively find all files with matching extensions in given directories."""
    files = []
    print("Scanning directories...", file=sys.stderr)
    
    total_scanned = 0
    # Use tqdm for a spinner/counter effect during scanning
    with tqdm(desc="Scanning files", unit=" items") as pbar:
        for directory in directories:
            path = Path(directory)
            if not path.exists():
                logging.warning(f"Directory not found: {directory}")
                continue
            
            logging.info(f"Scanning {directory}...")
            
            # os.walk can be slow on large trees, so we just update the counter as we go
            for root, _, filenames in os.walk(path):
                # Skip hidden directories
                if any(part.startswith('.') for part in Path(root).parts):
                    continue
                    
                for filename in filenames:
                    total_scanned += 1
                    if total_scanned % 100 == 0:
                        pbar.update(100)
                        
                    if filename.startswith('.'):
                        continue
                        
                    file_path = Path(root) / filename
                    if file_path.suffix.lower() in extensions:
                        files.append(file_path)
        pbar.update(total_scanned % 100) # Final update
        
    return files

def read_file_content(file_path: Path) -> str:
    """Read content from a file, handling different encodings and types."""
    # Handle .docx
    if file_path.suffix.lower() == '.docx':
        if Docx2txtLoader:
            try:
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                return "\n\n".join([d.page_content for d in docs])
            except Exception as e:
                logging.warning(f"Error reading .docx {file_path}: {e}")
                return ""
        else:
            logging.warning("docx2txt not installed, skipping .docx file")
            return ""

    # Handle text files (.md, .txt)
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.warning(f"Error reading {file_path}: {e}")
            return ""
    
    logging.error(f"Failed to read {file_path} with any supported encoding")
    return ""

def main():
    parser = argparse.ArgumentParser(description="Index local documents for RAG")
    parser.add_argument("--reset", action="store_true", help="Reset validation database before indexing")
    args = parser.parse_args()

    # Initialize Embeddings
    logging.info(f"Connecting to Ollama at {config.OLLAMA_BASE_URL}...")
    try:
        embeddings = OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.EMBEDDING_MODEL
        )
    except Exception as e:
        logging.error(f"Failed to connect to Ollama: {e}")
        sys.exit(1)

    # Initialize Vector Store
    persist_dir = config.CHROMA_PERSIST_DIRECTORY
    if args.reset:
        if os.path.exists(persist_dir):
            import shutil
            logging.warning(f"Resetting database at {persist_dir}")
            shutil.rmtree(persist_dir)
        if os.path.exists(STATE_FILE):
             os.remove(STATE_FILE)

    logging.info(f"Using ChromaDB at {persist_dir}")
    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Load state
    state = load_state()
    
    # Find Files
    current_files = get_files_to_index(config.DOCUMENT_DIRECTORIES, config.SUPPORTED_EXTENSIONS)
    logging.info(f"Found {len(current_files)} files to scan")

    # Detect changes
    files_to_process = []
    files_seen = set()

    for file_path in current_files:
        str_path = str(file_path)
        files_seen.add(str_path)
        
        try:
            mtime = os.path.getmtime(file_path)
            
            # Check if new or modified
            if str_path not in state or state[str_path].get("mtime") != mtime:
                files_to_process.append(file_path)
        except OSError:
            logging.warning(f"Could not access {file_path}")

    # Detect deletions
    files_to_delete = [f for f in state.keys() if f not in files_seen]

    logging.info(f"Processing plan: {len(files_to_process)} to index/update, {len(files_to_delete)} to delete")

    if not files_to_process and not files_to_delete:
        logging.info("No changes detected. Index is up to date.")
        return

    # Process Deletions
    if files_to_delete:
        logging.info(f"Removing {len(files_to_delete)} deleted files from index...")
        for file_path in files_to_delete:
            chunk_ids = state[file_path].get("chunk_ids", [])
            if chunk_ids:
                try:
                    vectorstore.delete(ids=chunk_ids)
                except Exception as e:
                    logging.error(f"Error deleting chunks for {file_path}: {e}")
            del state[file_path]
        save_state(state)

    # Process Updates/Additions
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    batch_size = 100
    documents_batch = []
    ids_batch = []
    metadata_batch = [] # Store tuple of (filepath, mtime, chunk_ids_for_this_file) to update state after batch
    
    # Create a mapping to easily update state after batch commit
    file_chunk_map = {} # filepath -> [new_chunk_ids]

    logging.info("Starting indexing process...")
    
    pbar = tqdm(files_to_process, desc="Indexing")
    for i, file_path in enumerate(pbar):
        # Update progress bar with current file name (truncated if long)
        pbar.set_postfix(file=file_path.name[-30:])
        
        str_path = str(file_path)
        content = read_file_content(file_path)
        
        # If existing file is being updated, remove old chunks first
        if str_path in state:
            old_ids = state[str_path].get("chunk_ids", [])
            if old_ids:
                try:
                    vectorstore.delete(ids=old_ids)
                except Exception as e:
                    logging.error(f"Error cleaning up old chunks for {str_path}: {e}")

        if not content or len(content.strip()) == 0:
            # Update state to track we saw it even if empty, so we don't re-process endlessly
            state[str_path] = {"mtime": os.path.getmtime(file_path), "chunk_ids": []}
            continue
            
        try:
            current_mtime = os.path.getmtime(file_path)
            chunks = text_splitter.create_documents(
                [content], 
                metadatas=[{
                    "source": str_path,
                    "filename": file_path.name,
                    "file_modified": time.ctime(current_mtime)
                }]
            )
            
            file_new_ids = []
            for j, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = j
                # Deterministic ID based on content hash or path+index would be ideal, 
                # but path+index+mtime ensures uniqueness on updates if we didn't delete first.
                # Since we delete first, path+index is fine, but let's make it robust.
                chunk_id = f"{file_path.name}_{current_mtime}_{j}"
                
                documents_batch.append(chunk)
                ids_batch.append(chunk_id)
                file_new_ids.append(chunk_id)

            file_chunk_map[str_path] = {"mtime": current_mtime, "chunk_ids": file_new_ids}

            # Batch processed
            if len(documents_batch) >= batch_size:
                vectorstore.add_documents(documents=documents_batch, ids=ids_batch)
                
                # Update state for files that were successfully committed
                for fpath, data in file_chunk_map.items():
                    state[fpath] = data
                save_state(state)
                
                documents_batch = []
                ids_batch = []
                file_chunk_map = {}
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    # Process remaining batch
    if documents_batch:
        vectorstore.add_documents(documents=documents_batch, ids=ids_batch)
        for fpath, data in file_chunk_map.items():
            state[fpath] = data
        save_state(state)
    
    logging.info("Indexing complete.")

if __name__ == "__main__":
    main()
