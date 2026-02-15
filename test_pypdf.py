from pypdf import PdfReader
from pathlib import Path

FILE_PATH = Path("/mnt/c/Users/wtrem/Downloads/2022-Return.pdf")

def main():
    print(f"Testing pypdf extraction for: {FILE_PATH}")
    
    if not FILE_PATH.exists():
        print("File not found!")
        return

    try:
        reader = PdfReader(FILE_PATH)
        print(f"Pages: {len(reader.pages)}")
        
        full_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
                
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
