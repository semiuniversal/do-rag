from docling.document_converter import DocumentConverter
from pathlib import Path

FILE_PATH = Path("/mnt/c/Users/wtrem/Downloads/2022-Return.pdf")

def main():
    print(f"Testing extraction for: {FILE_PATH}")
    
    if not FILE_PATH.exists():
        print("File not found!")
        return

    try:
        # Simplest possible invocation
        print("Initializing converter (defaults)...")
        converter = DocumentConverter()
        
        print("Converting...")
        result = converter.convert(FILE_PATH)
        markdown = result.document.export_to_markdown()
        
        print("\n--- Extracted Content Start ---")
        print(markdown[:500]) # First 500 chars
        print("--- Extracted Content End ---")
        
        if len(markdown.strip()) == 0:
            print("⚠️ WARNING: Extracted text is empty!")
        else:
            print(f"✅ Extracted {len(markdown)} characters.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
