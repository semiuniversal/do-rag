import asyncio
import hashlib
import time
import os
from pathlib import Path
from pypdf import PdfReader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_backend import get_embeddings
import config

TARGET_FILE = Path("/mnt/c/Users/wtrem/Downloads/2022-Return.pdf")

async def main():
    print(f"🚀 Force indexing: {TARGET_FILE}")
    
    # 1. Read
    content = ""
    try:
        reader = PdfReader(TARGET_FILE)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text + "\n"
        print(f"✅ Read {len(content)} chars.")
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return

    # 2. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    current_mtime = os.path.getmtime(TARGET_FILE)
    chunks = text_splitter.create_documents(
        [content],
        metadatas=[{
            "source": str(TARGET_FILE),
            "filename": TARGET_FILE.name,
            "file_modified": time.ctime(current_mtime),
        }]
    )
    print(f"✅ Created {len(chunks)} chunks.")

    # 3. Embed & Store
    print("Connecting to Qdrant...")
    embeddings = get_embeddings()
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        url=config.QDRANT_URL,
    )

    # Clean old chunks first (just in case)
    # We can't easily find old IDs without state file, but we can overwrite by ID if deterministic
    # But Qdrant overwrites by ID. Let's generate IDs deterministically as per indexer.py
    
    ids = []
    str_path = str(TARGET_FILE)
    for j, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = j
        id_str = f"{str_path}_{current_mtime}_{j}"
        chunk_id = hashlib.sha256(id_str.encode()).hexdigest()[:32]
        ids.append(chunk_id)

    print(f"Upserting {len(ids)} chunks to Qdrant...")
    vectorstore.add_documents(documents=chunks, ids=ids)
    print("✅ Upsert complete!")

    # 4. Update State File (Optional but polite)
    try:
        import json
        state_path = Path("indexing_state.json")
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
            state[str_path] = {"mtime": current_mtime, "chunk_ids": ids}
            with open(state_path, "w") as f:
                json.dump(state, f)
            print("✅ State file updated.")
    except Exception as e:
        print(f"⚠️ Failed to update state file: {e}")

if __name__ == "__main__":
    asyncio.run(main())
