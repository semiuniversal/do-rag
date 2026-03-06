
import asyncio
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_backend import get_embeddings
import config

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_large_file():
    file_path = "/mnt/c/Users/wtrem/Documents/com.pieces.os/production/Support/logs/log-05152025.txt"
    print(f"Reading {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    print(f"Content length: {len(content)} characters")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([content])
    print(f"Created {len(chunks)} chunks.")
    
    embeddings = get_embeddings()
    
    # Try embedding the first 5 chunks
    batch = chunks[:5]
    print("Embedding batch of 5...")
    try:
        vectors = embeddings.embed_documents([d.page_content for d in batch])
        print(f"Success! Embedded {len(vectors)} vectors.")
    except Exception as e:
        print(f"Embedding failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_large_file())
