import argparse
import sys
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb
import config

def search(query: str, top_k: int = config.TOP_K_RESULTS):
    """Search for documents matching the query."""
    
    # Initialize connection
    try:
        embeddings = OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.EMBEDDING_MODEL
        )
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        sys.exit(1)

    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )

    # Perform search
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    # Display results
    print(f"\nQuery: \"{query}\"\n")
    print("-" * 50)
    
    if not results:
        print("No results found.")
        return

    for i, (doc, score) in enumerate(results, 1):
        # Chroma returns distance, convert to pseudo-similarity if needed, 
        # or just display distance (lower is better for Euclidean/Cosine distance usually in Chroma default)
        # Note: Chroma default is L2 distance, lower = better. 
        metadata = doc.metadata
        print(f"Result {i} (Distance: {score:.4f})")
        print(f"File: {metadata.get('source', 'Unknown')}")
        print(f"Modified: {metadata.get('file_modified', 'Unknown')}")
        print("-" * 20)
        print(doc.page_content.strip())
        print("-" * 50)
        print()

def main():
    parser = argparse.ArgumentParser(description="Search local documents")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top_k", type=int, default=config.TOP_K_RESULTS, help="Number of results")
    args = parser.parse_args()

    if args.query:
        search(args.query, args.top_k)
    else:
        print("Interactive Search Mode (type 'exit' or 'quit' to stop)")
        while True:
            try:
                query = input("\nEnter query: ")
                if query.lower() in ('exit', 'quit'):
                    break
                if not query.strip():
                    continue
                search(query, args.top_k)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
