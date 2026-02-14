import argparse
import sys
from typing import List, Tuple
from langchain_qdrant import QdrantVectorStore
from llm_backend import get_embeddings
import config

def search(query: str, top_k: int = config.TOP_K_RESULTS):
    """Search for documents matching the query."""
    
    try:
        embeddings = get_embeddings()
    except Exception as e:
        print(f"Error connecting to embedding model: {e}")
        sys.exit(1)

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        url=config.QDRANT_URL,
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
        metadata = doc.metadata
        print(f"Result {i} (Score: {score:.4f})")
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
