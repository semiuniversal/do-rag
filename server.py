# Configure logging before any other imports
import logging
import sys
import logging_config
logging_config.setup_logging(also_stderr=False)  # Avoid breaking MCP stdout
logging.info("Starting server.py...")

try:
    from typing import List
    
    # Re-use existing search logic
    logging.info("Importing langchain...")
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    logging.info("Imported langchain")

    from llm_backend import get_embeddings, get_llm
    logging.info("Imported llm_backend")
    
    import config
    logging.info("Imported config")

    from mcp.server.fastmcp import FastMCP
    logging.info("Imported FastMCP")
except Exception as e:
    logging.error(f"Import failed: {e}")
    sys.exit(1)

try:
    mcp = FastMCP("Local RAG Search")
except Exception as e:
    logging.error(f"FastMCP init failed: {e}")
    sys.exit(1)

# --- Indexing & Config Tools ---

import asyncio
import settings as settings_module
from indexer import get_current_job, IndexingJob

_indexing_task = None  # holds the background asyncio.Task


@mcp.tool()
async def start_indexing(reset: bool = False) -> str:
    """
    Start indexing documents.
    """
    return "Please visit the Admin interface at http://localhost:5001 to manage indexing."


@mcp.tool()
async def get_indexing_status() -> str:
    """
    Get the current status of the indexing job.
    """
    return "Please visit the Admin interface at http://localhost:5001 to check indexing status."


@mcp.tool()
async def stop_indexing() -> str:
    """
    Stop the currently running indexing job.
    """
    return "Please visit the Admin interface at http://localhost:5001 to manage indexing."


@mcp.tool()
def list_config() -> str:
    """
    Show current configuration: indexed directories, excluded directories, and file extensions.
    """
    s = settings_module.load_settings()
    dirs = "\n".join(f"  • {d}" for d in s["directories"]) or "  (none configured)"
    excl = "\n".join(f"  • {e}" for e in s["exclusions"]) or "  (none)"
    exts = ", ".join(s["extensions"]) or "(none)"
    return f"Directories:\n{dirs}\n\nExclusions:\n{excl}\n\nFile types: {exts}"


@mcp.tool()
def add_directory(path: str) -> str:
    """Add a directory to be indexed. Use absolute paths."""
    from pathlib import Path
    if not Path(path).exists():
        return f"Warning: '{path}' does not exist. Added anyway — it will be skipped during indexing."
    result = settings_module.add_directory(path)
    return f"Directory added. Current directories:\n" + "\n".join(f"  • {d}" for d in result)


@mcp.tool()
def remove_directory(path: str) -> str:
    """Remove a directory from indexing."""
    result = settings_module.remove_directory(path)
    return f"Directory removed. Current directories:\n" + "\n".join(f"  • {d}" for d in result) if result else "No directories configured."


@mcp.tool()
def add_exclusion(pattern: str) -> str:
    """Add a directory name to exclude from indexing (e.g., '$Recycle.Bin', 'node_modules')."""
    result = settings_module.add_exclusion(pattern)
    return f"Exclusion added. Current exclusions:\n" + "\n".join(f"  • {e}" for e in result)


@mcp.tool()
def remove_exclusion(pattern: str) -> str:
    """Remove a directory exclusion pattern."""
    result = settings_module.remove_exclusion(pattern)
    return f"Exclusion removed. Current exclusions:\n" + "\n".join(f"  • {e}" for e in result) if result else "No exclusions configured."


@mcp.tool()
def add_extension(ext: str) -> str:
    """Add a file extension to index (e.g., '.py', '.pdf'). Leading dot is added if missing."""
    result = settings_module.add_extension(ext)
    return f"Extension added. Current extensions: {', '.join(result)}"


@mcp.tool()
def remove_extension(ext: str) -> str:
    """Remove a file extension from indexing."""
    result = settings_module.remove_extension(ext)
    return f"Extension removed. Current extensions: {', '.join(result)}" if result else "No extensions configured."


@mcp.tool()
def clear_index(confirm: bool = False) -> str:
    """
    Clear the entire document index. This is destructive and expensive to rebuild.
    First call without confirm to see what would be deleted.
    Call with confirm=True to actually clear.
    """
    import json
    from indexer import STATE_FILE

    # Check Qdrant collection
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        collection_info = client.get_collection(config.COLLECTION_NAME)
        point_count = collection_info.points_count or 0
    except Exception:
        point_count = 0

    has_state = STATE_FILE.exists()
    file_count = 0
    if has_state:
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            file_count = len(state)
        except Exception:
            pass

    if point_count == 0 and not has_state:
        return "Index is already empty."

    if not confirm:
        return (
            f"⚠️ This will permanently delete the index containing "
            f"{file_count} files ({point_count} chunks). "
            f"Rebuilding is expensive. "
            f"To proceed, call clear_index with confirm=True."
        )

    # Actually clear
    cleared = []
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        client.delete_collection(config.COLLECTION_NAME)
        cleared.append("Qdrant collection")
    except Exception as e:
        logging.warning(f"Could not delete Qdrant collection: {e}")
    if has_state:
        STATE_FILE.unlink()
        cleared.append("indexing state")

    return f"Index cleared ({', '.join(cleared)}). {file_count} files and {point_count} chunks removed. Please visit the Admin interface at http://localhost:5001 to rebuild."


@mcp.tool()
def show_index_stats() -> str:
    """
    Show statistics about the current document index: number of files indexed,
    total chunks, and a list of indexed files.
    """
    import json
    from indexer import STATE_FILE

    # Get chunk count from Qdrant
    qdrant_chunks = 0
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        info = client.get_collection(config.COLLECTION_NAME)
        qdrant_chunks = info.points_count or 0
    except Exception:
        pass

    if not STATE_FILE.exists():
        if qdrant_chunks > 0:
            return f"Qdrant has {qdrant_chunks} chunks but no state file. Consider re-indexing."
        return "No index found. Please visit the Admin interface at http://localhost:5001 to create one."

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except Exception as e:
        return f"Error reading index state: {e}"

    if not state:
        return "Index is empty. Please visit the Admin interface at http://localhost:5001 to index documents."

    total_files = len(state)

    # List files (truncate if too many)
    file_list = sorted(state.keys())
    if len(file_list) > 20:
        shown = "\n".join(f"  • {f}" for f in file_list[:20])
        shown += f"\n  ... and {len(file_list) - 20} more"
    else:
        shown = "\n".join(f"  • {f}" for f in file_list)

    return f"Index contains {total_files} files ({qdrant_chunks} chunks in Qdrant).\n\nFiles:\n{shown}"


@mcp.tool()
def system_stats() -> str:
    """
    Show system resource usage: RAM, VRAM (GPU), disk space, and CPU load
    for the do-rag application processes.
    """
    import subprocess
    import shutil
    import os

    lines = []

    # --- RAM ---
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                meminfo[parts[0].rstrip(":")] = int(parts[1])  # in kB
        total_gb = meminfo["MemTotal"] / 1048576
        avail_gb = meminfo["MemAvailable"] / 1048576
        used_gb = total_gb - avail_gb
        lines.append(f"RAM: {used_gb:.1f} / {total_gb:.1f} GB ({100 * used_gb / total_gb:.0f}% used)")
    except Exception as e:
        lines.append(f"RAM: unavailable ({e})")

    # --- VRAM (GPU) ---
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,gpu_name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            used_mb, total_mb = int(parts[0]), int(parts[1])
            gpu_name = parts[2] if len(parts) > 2 else "GPU"
            lines.append(f"VRAM ({gpu_name}): {used_mb} / {total_mb} MB ({100 * used_mb // total_mb}% used)")
        else:
            lines.append("VRAM: nvidia-smi error")
    except FileNotFoundError:
        lines.append("VRAM: nvidia-smi not found")
    except Exception as e:
        lines.append(f"VRAM: unavailable ({e})")

    # --- Disk ---
    try:
        # Project directory
        project_dir = str(config.PROJECT_ROOT)
        disk = shutil.disk_usage(project_dir)
        lines.append(f"Disk: {disk.used / (1024**3):.1f} / {disk.total / (1024**3):.0f} GB ({100 * disk.used // disk.total}% used)")

        # Index via Qdrant
        # Index via Qdrant
        try:
            # Check local state file for expected count
            from indexer import _load_state
            state = _load_state()
            local_count = len(state)
            
            # Check actual Qdrant collection
            qclient = QdrantClient(url=config.QDRANT_URL, timeout=3)
            try:
                info = qclient.get_collection(config.COLLECTION_NAME)
                lines.append(f"Qdrant index: {info.points_count} chunks | Local state: {local_count} files")
                if local_count > 0 and info.points_count == 0:
                    lines.append("WARNING: Index Integrity Mismatch! Local state has files but Qdrant is empty.")
                    lines.append("SUGGESTION: Visit http://localhost:5001 to rebuild index")
            except Exception:
                lines.append(f"Qdrant index: Collection not found | Local state: {local_count} files")
                lines.append("WARNING: Collection missing!")

            # File Distribution Analytics
            if local_count > 0:
                from collections import Counter
                from pathlib import Path
                extensions = []
                for path_str in state.keys():
                    ext = Path(path_str).suffix.lower() or "no_ext"
                    extensions.append(ext)
                
                counts = Counter(extensions)
                # Sort by count desc
                dist_str = ", ".join([f"{k}: {v}" for k, v in counts.most_common()])
                lines.append(f"Indexed Types: {dist_str}")

        except Exception as e:
            lines.append(f"Qdrant index: unavailable ({e})")
    except Exception as e:
        lines.append(f"Disk: unavailable ({e})")

    # --- CPU ---
    try:
        load1, load5, load15 = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        lines.append(f"CPU: {load1:.1f} / {load5:.1f} / {load15:.1f} (1/5/15 min avg, {cpu_count} cores)")
    except Exception as e:
        lines.append(f"CPU: unavailable ({e})")

    # --- Ollama models loaded ---
    try:
        import httpx
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/ps", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                model_lines = []
                for m in models:
                    name = m.get("name", "unknown")
                    size_gb = m.get("size", 0) / (1024**3)
                    model_lines.append(f"  • {name} ({size_gb:.1f} GB)")
                lines.append("Ollama loaded models:\n" + "\n".join(model_lines))
            else:
                lines.append("Ollama: no models loaded")
    except Exception:
        lines.append("Ollama: not reachable")

    return "\n".join(lines)


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Search local documents for the given query. 
    Returns relevant excerpts from markdown, text, and docx files.
    """
    logging.info(f"Searching for: {query}")
    
    try:
        embeddings = get_embeddings()
        
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=config.COLLECTION_NAME,
            url=config.QDRANT_URL,
        )

        results = vectorstore.similarity_search_with_score(query, k=top_k)
        
        if not results:
            return "No matching documents found."

        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            modified = doc.metadata.get('file_modified', 'Unknown')
            content = doc.page_content.strip()
            
            entry = f"""
Source: {source}
Modified: {modified}
Relevance Score: {score:.4f}
---
{content}
---
"""
            formatted_results.append(entry)

        return "\n".join(formatted_results)

    except Exception as e:
        logging.error(f"Search failed: {e}")
        return f"Error performing search: {str(e)}"

@mcp.tool()
def ask_documents(query: str) -> str:
    """
    Ask a question about the local documents. 
    Uses RAG (Retrieval Augmented Generation) to answer based on indexed content.
    """
    logging.info(f"Asking: {query}")
    
    try:
        # Initialize Embeddings
        embeddings = get_embeddings()
        
        # Initialize Vector Store
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=config.COLLECTION_NAME,
            url=config.QDRANT_URL,
        )
        
        # Initialize LLM
        llm = get_llm(temperature=0.3)
        
        # Create Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Define Prompt
        template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        # Define Chain
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
            
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Execute Chain
        response = chain.invoke(query)
        return response

    except Exception as e:
        logging.error(f"Ask failed: {e}")
        return f"Error generating answer: {str(e)}"

if __name__ == "__main__":
    import sys
    import uvicorn

    if "--transport" in sys.argv and ("sse" in sys.argv or "streamable-http" in sys.argv):
        # Monkeypatch TransportSecurityMiddleware to bypass host validation
        # Required for Docker containers (host.docker.internal) to connect
        try:
            from mcp.server.transport_security import TransportSecurityMiddleware
            
            async def noop_validate(self, request, is_post=False):
                return None
                
            TransportSecurityMiddleware.validate_request = noop_validate
            logging.info("Bypassed TransportSecurityMiddleware for Docker compatibility")
        except Exception as e:
            logging.warning(f"Failed to patch TransportSecurityMiddleware: {e}")

        # Dual transport: serve both SSE and Streamable HTTP on one port
        # Streamable HTTP must be the primary app (its lifespan initializes the task group)
        # SSE is mounted as a sub-app at /sse-transport
        primary_app = mcp.streamable_http_app()
        sse_app = mcp.sse_app()
        
        # Add SSE routes to the primary app
        from starlette.routing import Mount
        primary_app.routes.insert(0, Mount("/sse-transport", app=sse_app))

        logging.info("Starting dual-transport server:")
        logging.info("  Streamable HTTP: http://0.0.0.0:8000/mcp  (for Open WebUI)")
        logging.info("  SSE:             http://0.0.0.0:8000/sse-transport/sse  (for IDEs)")
        uvicorn.run(primary_app, host="0.0.0.0", port=8000)
    else:
        # stdio mode for direct IDE integration (Claude Desktop, etc.)
        mcp.run()
