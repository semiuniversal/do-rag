from mcp.server.fastmcp import FastMCP
from typing import List
import sys
import logging

# Re-use existing search logic
# We need to suppress print statements from search_docs if we import it, 
# or better yet, refactor search_docs to separate logic from printing.
# For now, let's copy the core search logic to avoid stdout pollution which breaks MCP.

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import config

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
    mcp.run()
