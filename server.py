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
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    logging.info("Imported langchain")
    
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

@mcp.tool()
def ask_documents(query: str) -> str:
    """
    Ask a question about the local documents. 
    Uses RAG (Retrieval Augmented Generation) to answer based on indexed content.
    """
    logging.info(f"Asking: {query}")
    
    try:
        # Initialize Embeddings
        embeddings = OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.EMBEDDING_MODEL
        )
        
        # Initialize Vector Store
        vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        
        # Initialize LLM
        llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.LLM_MODEL,
            temperature=0.3
        )
        
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
