
import asyncio
from pypdf import PdfReader
from llm_backend import get_embeddings
import logging

logging.basicConfig(level=logging.INFO)

def test_pdf():
    file_path = "/mnt/c/Users/wtrem/Downloads/2024-Return.pdf"
    print(f"Reading {file_path}...")
    try:
        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"
        print(f"Extracted {len(content)} characters.")
        
        embeddings = get_embeddings()
        vector = embeddings.embed_query(content[:1000]) # Test embedding first chunk
        print(f"Success! Vector len: {len(vector)}")
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_pdf()
