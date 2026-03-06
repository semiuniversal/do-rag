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
from typing import Any, Dict, List, Optional, Tuple

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
    """Read content from a file using Docling for rich formats, or standard I/O for text."""
    suffix = file_path.suffix.lower()
    
    # Fast path for plain text and code
    # HTML could go either way, but Docling does infinite better job than raw read
    if suffix in [".md", ".txt", ".py", ".js", ".json", ".sh", ".css", ".sql", ".yaml", ".yml"]:
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
        return ""

    # Priority: Use pypdf for PDF files to avoid docling hangs/crashes
    content = ""
    if file_path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            logging.info(f"Using pypdf for {file_path}")
            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
            return content
        except ImportError:
            logging.warning("pypdf not installed. Falling back to docling.")
        except Exception as e:
            logging.warning(f"pypdf failed for {file_path}: {e}")

    # Docling path for DOCX, PPTX, XLSX, HTML (and PDF fallback if pypdf failed)
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice

        # Configure for CPU only to save VRAM for Ollama
        accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        # Initialize converter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
                InputFormat.DOCX: pipeline_options,
                InputFormat.PPTX: pipeline_options,
                InputFormat.XLSX: pipeline_options,
                InputFormat.HTML: pipeline_options
            }
        )
        result = converter.convert(file_path)
        content = result.document.export_to_markdown()
        
    except ImportError:
        logging.warning("Docling not installed.")
    except Exception as e:
        logging.warning(f"Docling failed for {file_path}: {e}")

    return content


def _scan_files(directories: List[str], extensions: List[str], exclusions: List[str]) -> List[Tuple[Path, float]]:
    """Recursively find all files with matching extensions, respecting exclusions.
    Returns list of (Path, mtime) tuples.
    """
    files: List[Tuple[Path, float]] = []
    
    # Pre-process exclusions for faster lookup
    # sets are faster than lists
    exclude_names = set(exclusions)
    
    # Create a thread pool for parallel scanning
    # Limit max workers to avoid excessive threads, but enough to cover I/O latency
    import concurrent.futures
    
    # Check for Windows Optimization
    user_settings = settings_module.load_settings()
    if user_settings.get("optimize_for_windows", False):
         try:
             import windows.bridge as win_bridge
             logging.info("Attempting Windows Native Scan...")
             bridge_files = win_bridge.scan_windows(directories, extensions, list(exclude_names))
             if bridge_files is not None:
                 logging.info(f"Windows Native Scan successful: {len(bridge_files)} files.")
                 return bridge_files
             else:
                 logging.warning("Windows Native Scan returned None, falling back to parallel scan.")
         except Exception as e:
             logging.error(f"Failed to use Windows Native Scan: {e}")

    max_workers = min(32, (os.cpu_count() or 1) * 4) 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map of future -> directory path
        futures = {}
        
        # Helper to submit a directory scan task
        def submit_scan(dir_path: Path):
            return executor.submit(_scan_directory_task, dir_path, exclude_names, extensions)

        # Initial submissions
        for directory in directories:
            path = Path(directory)
            if not path.exists():
                logging.warning(f"Directory not found: {directory}")
                continue
            futures[submit_scan(path)] = path
            
        while futures:
            # Wait for any future to complete
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            
            for future in done:
                path = futures.pop(future)
                try:
                    result_files, result_subdirs = future.result()
                    files.extend(result_files)
                    
                    # Submit subdirectories
                    for subdir in result_subdirs:
                         futures[submit_scan(subdir)] = subdir
                         
                except Exception as e:
                    logging.warning(f"Error scanning {path}: {e}")

    return files

def _scan_directory_task(path: Path, exclude_names: set, extensions: List[str]) -> Tuple[List[Tuple[Path, float]], List[Path]]:
    """
    Scans a single directory. Returns (files, subdirectories).
    Executed in a thread pool.
    """
    local_files = []
    subdirs = []
    try:
        with os.scandir(path) as it:
            # Consume iterator
            entries = list(it)
            
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                
                if entry.is_dir():
                    if entry.name not in exclude_names:
                        subdirs.append(Path(entry.path))
                    continue

                # File processing
                if (entry.name.endswith(".log") or 
                    entry.name.endswith(".log.txt") or 
                    "_logs" in entry.name or 
                    entry.name.startswith("log-")):
                    continue

                _, ext = os.path.splitext(entry.name)
                if ext.lower() not in extensions:
                        continue

                try:
                    stat = entry.stat()
                    if stat.st_size > 5 * 1024 * 1024:
                            continue
                            
                    local_files.append((Path(entry.path), stat.st_mtime))
                except OSError:
                    pass
                    
    except OSError as e:
        # Don't log here to avoid thread contention on logging lock? 
        # Or just let the main thread handle exception?
        # Re-raising allows main loop to log associated with path
        raise e
        
    return local_files, subdirs


def unload_llm():
    """Explicitly unload the chat model to free VRAM for embeddings."""
    try:
        import requests
        import config
        # Use simple requests to avoid circular dependencies or complex client init
        # Payload keep_alive: 0 triggers unload
        model = config.LLM_MODEL
        logging.info(f"Unloading LLM {model} to free VRAM...")
        try:
             # Try /api/generate (for older ollama) or /api/chat
             requests.post(f"{config.OLLAMA_BASE_URL}/api/generate", 
                           json={"model": model, "keep_alive": 0}, timeout=2)
        except Exception:
             pass
    except Exception as e:
        logging.warning(f"Failed to unload LLM: {e}")

class IndexingJob:
    """Manages a single indexing run with progress tracking and cancellation."""

    def __init__(self):
        self.status: str = "idle"  # idle | preparing | scanning | indexing | complete | cancelled | error
        self.total_files: int = 0
        self.files_to_process: int = 0
        self.processed_files: int = 0
        self.current_file: str = ""
        self.errors: List[str] = []
        self.files_deleted: int = 0
        self._cancelled: bool = False
        self.detail_status: str = ""  # Granular status message (e.g. "Loading models...")
        self._start_time: Optional[float] = None

    def cancel(self):
        """Request cancellation of the current job."""
        if self.status in ("preparing", "scanning", "indexing"):
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
            "detail_status": self.detail_status,
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
        if self.status in ("preparing", "scanning", "indexing"):
            return "An indexing job is already running. Use stop_indexing() first."

        self._cancelled = False
        self.errors = []
        self.processed_files = 0
        self.files_deleted = 0
        self.current_file = ""

        try:
            self.status = "preparing"
            self.detail_status = "Loading settings..."
            logging.info("Indexing job started: Preparing...")

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
                self.detail_status = "Checking active models..."
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
                                self.detail_status = f"Unloading LLM {name}..."
                                logging.info(f"Unloading LLM {name} to free VRAM for indexing...")
                                await client.post(f"{config.OLLAMA_BASE_URL}/api/generate", 
                                                json={"model": name, "keep_alive": 0})
            except Exception as e:
                logging.warning(f"Failed to unload other models: {e}")

            # Import heavy deps
            self.detail_status = "Importing heavy libraries (LangChain, Qdrant)..."
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from llm_backend import get_embeddings

            # Unload LLM to prevent VRAM contention
            self.detail_status = "Unloading Chat LLM..."
            unload_llm()

            self.detail_status = "Initializing Embeddings..."
            embeddings = get_embeddings()
            if hasattr(embeddings, "base_url"):
                 logging.info(f"Indexer using embedding base_url: {embeddings.base_url}")


            # Handle reset or integrity check
            self.detail_status = "Checking Index Integrity..."
            if reset:
                try:
                    client = QdrantClient(url=config.QDRANT_URL)
                    client.delete_collection(config.COLLECTION_NAME)
                    logging.warning(f"Deleted Qdrant collection: {config.COLLECTION_NAME}")
                except Exception:
                    pass  # collection may not exist yet
                if STATE_FILE.exists():
                    STATE_FILE.unlink()
            else:
                # Integrity Check: If state exists but Qdrant is missing/empty, force partial reset
                try:
                    client = QdrantClient(url=config.QDRANT_URL)
                    try:
                        info = client.get_collection(config.COLLECTION_NAME)
                        if info.points_count == 0 and STATE_FILE.exists():
                            logging.warning("Index integrity check failed: Qdrant empty but state file exists. Forcing re-index.")
                            STATE_FILE.unlink()
                    except Exception:
                        logging.warning("Index integrity check failed: Qdrant collection missing. Forcing re-index.")
                        if STATE_FILE.exists():
                            STATE_FILE.unlink()
                except Exception as e:
                    logging.error(f"Failed to perform integrity check: {e}")

            # Scan files
            self.status = "scanning"
            self.detail_status = "Scanning files..."
            scan_start = time.time()
            logging.info("Starting file scan...")
            all_files = await loop.run_in_executor(
                None, _scan_files, directories, extensions, exclusions
            )
            scan_duration = time.time() - scan_start
            logging.info(f"File scan completed in {scan_duration:.4f}s. Found {len(all_files)} files.")
            
            self.total_files = len(all_files)

            if self._cancelled:
                return self.get_status_text()

            # Load state and detect changes
            state = _load_state()
            files_to_process = []
            files_seen = set()

            for file_path, mtime in all_files:
                str_path = str(file_path)
                files_seen.add(str_path)
                try:
                    # mtime is already retrieved from scan
                    if str_path not in state or state[str_path].get("mtime") != mtime:
                        files_to_process.append(file_path)
                except OSError:
                    logging.warning(f"Could not access {file_path}")

            # Detect deletions
            files_to_delete = [f for f in state.keys() if f not in files_seen]
            self.files_to_process = len(files_to_process)

            if not files_to_process and not files_to_delete:
                _save_state(state)  # Update state timestamp to acknowledge current settings
                self.status = "complete"
                return f"No changes detected. Index is up to date ({self.total_files} files)."

            self.status = "indexing"
            logging.info(f"Starting indexing of {self.files_to_process} files (modified or new).")
            # _start_time is already set above

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
                    # Adaptive Batching: Try desired batch size, split on failure
                    target_batch_size = getattr(config, "INDEXING_BATCH_SIZE", 10)
                    chunk_limit = getattr(config, "INDEXING_MAX_BATCH_TOKENS", 8000)

                    # Helper function for recursive adaptive batching
                    async def _adaptive_embed(docs_subset, ids_subset, depth=0):
                        if not docs_subset:
                            return

                        # 1. Check character limit first (simple heuristic for tokens)
                        total_chars = sum(len(d.page_content) for d in docs_subset)
                        # If over limit and we have more than 1 item, split immediately
                        if total_chars > chunk_limit and len(docs_subset) > 1:
                            mid = len(docs_subset) // 2
                            await _adaptive_embed(docs_subset[:mid], ids_subset[:mid], depth+1)
                            await _adaptive_embed(docs_subset[mid:], ids_subset[mid:], depth+1)
                            return

                        # 2. Try to embed current batch
                        try:
                            # Small delay before every request to be kind to the LLM
                            batch_delay = getattr(config, "INDEXING_DELAY", 0.1)
                            if batch_delay > 0:
                                await asyncio.sleep(batch_delay)

                            await loop.run_in_executor(
                                None,
                                lambda d=docs_subset, ids=ids_subset: vectorstore.add_documents(
                                    documents=d, ids=ids
                                ),
                            )
                            # Success!
                            return 

                        except Exception as e:
                            err_msg = str(e)
                            # If batch size is 1, we can't split further. Retrying with backoff is the only option.
                            if len(docs_subset) == 1:
                                if "EOF" in err_msg or "500" in err_msg or "Connection refused" in err_msg:
                                    # Serious error on single item -> Backoff retry
                                    max_retries = 3
                                    for attempt in range(max_retries):
                                        wait_time = 2 * (2 ** attempt) # 2, 4, 8
                                        logging.warning(f"Single item embedding failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s: {e}")
                                        await asyncio.sleep(wait_time)
                                        try:
                                            await loop.run_in_executor(
                                                None,
                                                lambda d=docs_subset, ids=ids_subset: vectorstore.add_documents(
                                                    documents=d, ids=ids
                                                ),
                                            )
                                            return # Success on retry
                                        except Exception as final_e:
                                            if attempt == max_retries - 1:
                                                raise final_e
                                else:
                                    raise e
                            else:
                                # Batch > 1 failed -> Split and recurse (Divide and Conquer)
                                logging.warning(f"Batch of {len(docs_subset)} items failed (chars={total_chars}). Splitting and retrying...")
                                mid = len(docs_subset) // 2
                                await _adaptive_embed(docs_subset[:mid], ids_subset[:mid], depth+1)
                                await _adaptive_embed(docs_subset[mid:], ids_subset[mid:], depth+1)

                    # Execute adaptive batching for the whole file's chunks
                    # We still chunk the initial loop by target_batch_size to avoid passing 1000 items to recursion start
                    for i in range(0, len(chunks), target_batch_size):
                        batch_docs = chunks[i:i + target_batch_size]
                        batch_ids = file_new_ids[i:i + target_batch_size]
                        
                        await _adaptive_embed(batch_docs, batch_ids)

                    state[str_path] = {"mtime": current_mtime, "chunk_ids": file_new_ids}
                    files_since_save += 1

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
