
import sys
import logging
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llm_backend import get_embeddings
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_stack():
    print("=== Core Stack Verification ===")
    
    # 1. Test Embedding Connection
    print("[1/3] Testing Embedding Model...")
    try:
        embeddings = get_embeddings()
        vector = embeddings.embed_query("Sanity check")
        if len(vector) > 0:
            print(f"  Success! Vector length: {len(vector)}")
        else:
            print("  Failure: Empty vector returned.")
            return False
    except Exception as e:
        print(f"  Failure: {e}")
        return False

    # 2. Test Qdrant Connection
    print("[2/3] Testing Qdrant Connection...")
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        info = client.get_collections()
        print(f"  Success! Connected to Qdrant. Collections: {[c.name for c in info.collections]}")
    except Exception as e:
        print(f"  Failure: {e}")
        return False
        
    # 3. Test Upsert & Search
    print("[3/3] Testing Upsert & Search...")
    try:
        # Create a temporary collection for verification
        test_collection = "verify_stack_test"
        client.recreate_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=len(vector), distance=Distance.COSINE),
        )
        
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=test_collection,
            embedding=embeddings,
        )
        
        text = "The quick brown fox jumps over the lazy dog."
        vectorstore.add_texts([text], metadatas=[{"id": "test_1"}])
        
        results = vectorstore.similarity_search("brown fox", k=1)
        if results and results[0].page_content == text:
            print("  Success! Retrieved document matches.")
        else:
            print(f"  Failure: Search returned unexpected results: {results}")
            return False
            
        # Cleanup
        client.delete_collection(test_collection)
        
    except Exception as e:
        print(f"  Failure: {e}")
        return False

    print("\n=== VERIFICATION COMPLETE: ALL SYSTEMS GO ===")
    return True

if __name__ == "__main__":
    success = verify_stack()
    sys.exit(0 if success else 1)
