"""Test PDF extraction with PyMuPDF (fitz)."""
import fitz
from pathlib import Path

FILE_PATH = Path("/mnt/c/Users/wtrem/Downloads/2022-Return.pdf")

def main():
    print(f"Testing PyMuPDF extraction for: {FILE_PATH}")
    
    if not FILE_PATH.exists():
        print("File not found!")
        return

    try:
        with fitz.open(FILE_PATH) as doc:
            print(f"Pages: {len(doc)}")
            parts = [page.get_text(sort=True) for page in doc if page.get_text(sort=True)]
            full_text = "\n\n".join(parts)
                
        print("\n--- Extracted Content Start ---")
        print(full_text[:500]) 
        print("--- Extracted Content End ---")
        
        if len(full_text.strip()) == 0:
            print("⚠️ WARNING: Extracted text is empty!")
        else:
            print(f"✅ Extracted {len(full_text)} characters.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
