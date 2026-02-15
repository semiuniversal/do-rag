import config
from qdrant_client import QdrantClient
import hashlib
import os

def main():
    client = QdrantClient(url=config.QDRANT_URL)
    
    # We need to find points where 'source' metadata matches the file path
    # But Qdrant filtering is better than guessing IDs
    
    target_file = "/mnt/c/Users/wtrem/Downloads/2022-Return.pdf"
    print(f"Inspecting chunks for: {target_file}")
    
    try:
        # Search by filter
        scroll_result, _ = client.scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter={
                "must": [
                    {"key": "metadata.source", "match": {"value": target_file}}
                ]
            },
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        if not scroll_result:
            print("❌ No chunks found in Qdrant for this file!")
            return
            
        print(f"✅ Found {len(scroll_result)} chunks.")
        for point in scroll_result:
            payload = point.payload
            content = payload.get("page_content", "")
            print("-" * 40)
            print(f"Chunk ID: {point.id}")
            print(f"Length: {len(content)} chars")
            print(f"Snippet: {content[:200]}...") 
            print("-" * 40)
            
    except Exception as e:
        print(f"Error querying Qdrant: {e}")

if __name__ == "__main__":
    main()
