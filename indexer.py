# indexer.py
# Async, cancellable indexing engine for the MCP server.
# Extracted from index_docs.py logic, designed to run as a background task.

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import settings as settings_module


STATE_FILE = Path(__file__).parent / "indexing_state.json"


def _load_state() -> Dict[str, Dict[str, Any]]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load state file: {e}")
    return {}


def _save_state(state: Dict[str, Dict[str, Any]]):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _read_file_content(file_path: Path) -> str:
    """Read content from a file, handling different encodings and types."""
    # Handle .docx
    if file_path.suffix.lower() == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
            return "\n\n".join([d.page_content for d in docs])
        except ImportError:
            logging.warning("docx2txt not installed, skipping .docx file")
            return ""
        except Exception as e:
            logging.warning(f"Error reading .docx {file_path}: {e}")
            return ""

    # Handle text files (.md, .txt, code, etc.)
    encodings = ["utf-8", "latin-1", "cp1252"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.warning(f"Error reading {file_path}: {e}")
            return ""

    logging.error(f"Failed to read {file_path} with any supported encoding")
    return ""


def _scan_files(directories: List[str], extensions: List[str], exclusions: List[str]) -> List[Path]:
    """Recursively find all files with matching extensions, respecting exclusions."""
    files = []
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            logging.warning(f"Directory not found: {directory}")
            continue

        for root, dirs, filenames in os.walk(path):
            # Prune excluded and hidden directories in-place
            dirs[:] = [d for d in dirs if d not in exclusions and not d.startswith(".")]

            if any(part.startswith(".") for part in Path(root).parts):
                continue

            for filename in filenames:
                if filename.startswith("."):
                    continue
                file_path = Path(root) / filename
                if file_path.suffix.lower() in extensions:
                    files.append(file_path)

    return files


class IndexingJob:
    """Manages a single indexing run with progress tracking and cancellation."""

    def __init__(self):
        self.status: str = "idle"  # idle | scanning | indexing | complete | cancelled | error
        self.total_files: int = 0
        self.files_to_process: int = 0
        self.processed_files: int = 0
        self.current_file: str = ""
        self.errors: List[str] = []
        self.files_deleted: int = 0
        self._cancelled: bool = False
        self._start_time: Optional[float] = None

    def cancel(self):
        """Request cancellation of the current job."""
        if self.status in ("scanning", "indexing"):
            self._cancelled = True
            self.status = "cancelled"

    def get_status(self) -> dict:
        """Return current job status as a dict."""
        result = {
            "status": self.status,
            "total_files_found": self.total_files,
            "files_to_process": self.files_to_process,
            "processed_files": self.processed_files,
            "current_file": self.current_file,
            "errors": len(self.errors),
            "files_deleted": self.files_deleted,
        }
        if self._start_time and self.status == "indexing" and self.files_to_process > 0:
            elapsed = time.time() - self._start_time
            if self.processed_files > 0:
                rate = elapsed / self.processed_files
                remaining = (self.files_to_process - self.processed_files) * rate
                result["elapsed_seconds"] = round(elapsed)
                result["eta_seconds"] = round(remaining)
                result["percent"] = round(100 * self.processed_files / self.files_to_process)
        return result

    def get_status_text(self) -> str:
        """Return a human-readable progress string."""
        s = self.get_status()
        if s["status"] == "idle":
            return "No indexing job running."
        if s["status"] == "scanning":
            return "Scanning directories for files..."
        if s["status"] == "cancelled":
            return f"Indexing cancelled. Processed {s['processed_files']}/{s['files_to_process']} files."
        if s["status"] == "error":
            return f"Indexing failed: {self.errors[-1] if self.errors else 'unknown error'}"
        if s["status"] == "complete":
            return (
                f"Indexing complete. {s['processed_files']} files indexed, "
                f"{s['files_deleted']} deleted from index. "
                f"{s['errors']} errors."
            )
        # indexing
        pct = s.get("percent", 0)
        eta = s.get("eta_seconds", 0)
        eta_str = f"{eta // 60}m {eta % 60}s" if eta else "calculating..."
        return (
            f"Indexing: {s['processed_files']}/{s['files_to_process']} files "
            f"({pct}%) — ETA: {eta_str}\n"
            f"Current: {s['current_file']}"
        )

    async def start(self, reset: bool = False) -> str:
        """Run the indexing job. Returns a summary string when done."""
        if self.status in ("scanning", "indexing"):
            return "An indexing job is already running. Use stop_indexing() first."

        self._cancelled = False
        self.errors = []
        self.processed_files = 0
        self.files_deleted = 0
        self.current_file = ""

        try:
            self.status = "scanning"

            # Load settings
            user_settings = settings_module.load_settings()
            directories = user_settings["directories"]
            extensions = user_settings["extensions"]
            exclusions = user_settings["exclusions"]

            if not directories:
                self.status = "error"
                self.errors.append("No directories configured. Use add_directory() first.")
                return self.errors[-1]

            # Run blocking I/O in executor
            loop = asyncio.get_running_loop()

            # Attempt to unload other models to free VRAM
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/ps")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("models", [])
                        for m in models:
                            name = m["name"]
                            # precise check or loose check? 'nomic-embed-text' vs 'nomic-emb'
                            # If it's NOT the embedding model, unload it.
                            # config.EMBEDDING_MODEL might be 'nomic-embed-text'
                            # name might be 'nomic-embed-text:latest'
                            if config.EMBEDDING_MODEL not in name and "nomic" not in name:
                                logging.info(f"Unloading LLM {name} to free VRAM for indexing...")
                                await client.post(f"{config.OLLAMA_BASE_URL}/api/generate", 
                                                json={"model": name, "keep_alive": 0})
            except Exception as e:
                logging.warning(f"Failed to unload other models: {e}")

            # Import heavy deps
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from llm_backend import get_embeddings

            embeddings = get_embeddings()

            # Handle reset
            if reset:
                try:
                    client = QdrantClient(url=config.QDRANT_URL)
                    client.delete_collection(config.COLLECTION_NAME)
                    logging.warning(f"Deleted Qdrant collection: {config.COLLECTION_NAME}")
                except Exception:
                    pass  # collection may not exist yet
                if STATE_FILE.exists():
                    STATE_FILE.unlink()

            # Scan files
            all_files = await loop.run_in_executor(
                None, _scan_files, directories, extensions, exclusions
            )
            self.total_files = len(all_files)

            if self._cancelled:
                return self.get_status_text()

            # Load state and detect changes
            state = _load_state()
            files_to_process = []
            files_seen = set()

            for file_path in all_files:
                str_path = str(file_path)
                files_seen.add(str_path)
                try:
                    mtime = os.path.getmtime(file_path)
                    if str_path not in state or state[str_path].get("mtime") != mtime:
                        files_to_process.append(file_path)
                except OSError:
                    logging.warning(f"Could not access {file_path}")

            # Detect deletions
            files_to_delete = [f for f in state.keys() if f not in files_seen]
            self.files_to_process = len(files_to_process)

            if not files_to_process and not files_to_delete:
                self.status = "complete"
                return f"No changes detected. Index is up to date ({self.total_files} files)."

            self.status = "indexing"
            self._start_time = time.time()

            # Initialize vector store
            # Ensure collection exists
            client = QdrantClient(url=config.QDRANT_URL)
            if not client.collection_exists(config.COLLECTION_NAME):
                # Nomic embed dimension = 768
                client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
            vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=config.COLLECTION_NAME,
                url=config.QDRANT_URL,
            )

            # Process deletions
            if files_to_delete:
                for file_path in files_to_delete:
                    chunk_ids = state[file_path].get("chunk_ids", [])
                    if chunk_ids:
                        try:
                            await loop.run_in_executor(None, vectorstore.delete, chunk_ids)
                        except Exception as e:
                            logging.error(f"Error deleting chunks for {file_path}: {e}")
                    del state[file_path]
                    self.files_deleted += 1
                _save_state(state)

            if self._cancelled:
                return self.get_status_text()

            # Process new/modified files
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            batch_size = getattr(config, "INDEXING_BATCH_SIZE", 10)
            state_save_interval = 50  # save state every N files
            files_since_save = 0

            for file_path in files_to_process:
                if self._cancelled:
                    break

                str_path = str(file_path)
                self.current_file = file_path.name

                try:
                    content = await loop.run_in_executor(None, _read_file_content, file_path)

                    # Remove old chunks if updating
                    if str_path in state:
                        old_ids = state[str_path].get("chunk_ids", [])
                        if old_ids:
                            await loop.run_in_executor(None, vectorstore.delete, old_ids)

                    if not content or len(content.strip()) == 0:
                        state[str_path] = {"mtime": os.path.getmtime(file_path), "chunk_ids": []}
                        self.processed_files += 1
                        continue

                    current_mtime = os.path.getmtime(file_path)
                    chunks = text_splitter.create_documents(
                        [content],
                        metadatas=[{
                            "source": str_path,
                            "filename": file_path.name,
                            "file_modified": time.ctime(current_mtime),
                        }],
                    )

                    # Generate IDs for all chunks in this file
                    file_new_ids = []
                    for j, chunk in enumerate(chunks):
                        chunk.metadata["chunk_index"] = j
                        id_str = f"{str_path}_{current_mtime}_{j}"
                        chunk_id = hashlib.sha256(id_str.encode()).hexdigest()[:32]
                        file_new_ids.append(chunk_id)

                    # Batch-add chunks to ChromaDB/Qdrant
                    for i in range(0, len(chunks), batch_size):
                        batch_docs = chunks[i:i + batch_size]
                        batch_ids = file_new_ids[i:i + batch_size]
                        
                        # Retry logic for transient Ollama errors (EOF/500)
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                await loop.run_in_executor(
                                    None,
                                    lambda d=batch_docs, ids=batch_ids: vectorstore.add_documents(
                                        documents=d, ids=ids
                                    ),
                                )
                                break  # success
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    raise e  # re-raise last error
                                logging.warning(f"Batch index failed (attempt {attempt+1}/{max_retries}), retrying: {e}")
                                await asyncio.sleep(1 * (attempt + 1))  # linear backoff

                    state[str_path] = {"mtime": current_mtime, "chunk_ids": file_new_ids}
                    files_since_save += 1

                    # Save state periodically, not every file
                    if files_since_save >= state_save_interval:
                        _save_state(state)
                        files_since_save = 0

                except Exception as e:
                    error_msg = f"Error processing {file_path.name}: {e}"
                    logging.error(error_msg)
                    self.errors.append(error_msg)

                self.processed_files += 1

            if self._cancelled:
                _save_state(state)  # save progress before exit
                return self.get_status_text()

            _save_state(state)  # final save
            self.status = "complete"
            return self.get_status_text()

        except Exception as e:
            self.status = "error"
            error_msg = f"Indexing failed: {e}"
            self.errors.append(error_msg)
            logging.error(error_msg)
            return error_msg


# Singleton job instance for the MCP server
_current_job: Optional[IndexingJob] = None


def get_current_job() -> IndexingJob:
    """Get or create the singleton IndexingJob."""
    global _current_job
    if _current_job is None:
        _current_job = IndexingJob()
    return _current_job
