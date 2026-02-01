# Configure logging to stderr so it doesn't break MCP stdout transport
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logging.info("Starting server.py...")

try:
    from mcp.server.fastmcp import FastMCP
    logging.info("Imported FastMCP")
    from typing import List
    
    # Re-use existing search logic
    logging.info("Importing langchain...")
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    logging.info("Imported langchain")
    
    import config
    logging.info("Imported config")
except Exception as e:
    logging.error(f"Import failed: {e}")
    sys.exit(1)

# Configure logging to stderr so it doesn't break MCP stdout transport
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("Local RAG Search")

@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Search local documents for the given query. 
    Returns relevant excerpts from markdown, text, and docx files.
    """
    logging.info(f"Searching for: {query}")
    
    try:
        embeddings = OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.EMBEDDING_MODEL
        )
        
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

if __name__ == "__main__":
    # Force SSE mode for debugging
    logging.info("Calling mcp.run(transport='sse', host='0.0.0.0', port=8000)...")
    # Note: FastMCP.run() arguments might vary by version. 
    # If this fails, we will revert to CLI. 
    # But usually it passes kwargs to uvicorn.run
    try:
        mcp.settings.port = 8000
        mcp.settings.host = "0.0.0.0"
        mcp.run(transport='sse')
    except Exception as e:
        logging.error(f"Run failed: {e}")
        # Fallback for older SDK versions that might rely on CLI
        mcp.run()
