# Configure logging to stderr so it doesn't break MCP stdout transport
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logging.info("Starting server.py...")

try:
    from typing import List
    
    # Re-use existing search logic
    logging.info("Importing langchain...")
    from langchain_chroma import Chroma
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

# Configure logging to stderr so it doesn't break MCP stdout transport
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

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
    Start indexing documents in the background.
    Scans configured directories and indexes new or modified files.
    Set reset=True to clear the index and re-index everything.
    """
    global _indexing_task
    job = get_current_job()

    if job.status in ("scanning", "indexing"):
        return f"Indexing is already running. {job.get_status_text()}"

    # Create a fresh job
    from indexer import _current_job
    import indexer
    indexer._current_job = IndexingJob()
    job = indexer._current_job

    _indexing_task = asyncio.create_task(job.start(reset=reset))
    # Give it a moment to start scanning
    await asyncio.sleep(0.5)
    return job.get_status_text()


@mcp.tool()
async def get_indexing_status() -> str:
    """
    Get the current status of the indexing job.
    Shows progress, current file, ETA, and error count.
    """
    job = get_current_job()
    return job.get_status_text()


@mcp.tool()
async def stop_indexing() -> str:
    """
    Stop the currently running indexing job.
    Already-indexed files are kept in the index.
    """
    job = get_current_job()
    if job.status not in ("scanning", "indexing"):
        return "No indexing job is currently running."
    job.cancel()
    return "Indexing cancelled. " + job.get_status_text()


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
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Search local documents for the given query. 
    Returns relevant excerpts from markdown, text, and docx files.
    """
    logging.info(f"Searching for: {query}")
    
    try:
        embeddings = get_embeddings()
        
        vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
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
        vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
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
