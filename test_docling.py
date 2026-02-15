from docling.document_converter import DocumentConverter
from pathlib import Path

def test_conversion(file_path):
    print(f"Converting {file_path}...")
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        md_output = result.document.export_to_markdown()
        print("--- Markdown Output ---")
        print(md_output[:500] + "..." if len(md_output) > 500 else md_output)
        print("-----------------------")
        return True
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return False

if __name__ == "__main__":
    # Create a dummy HTML file for testing
    dummy_html = Path("test_docling.html")
    dummy_html.write_text("<html><body><h1>Hello Docling</h1><p>This is a test.</p></body></html>")
    
    if test_conversion(dummy_html):
        print("Docling test passed!")
    else:
        print("Docling test failed!")
    
    # Cleanup
    if dummy_html.exists():
        dummy_html.unlink()
