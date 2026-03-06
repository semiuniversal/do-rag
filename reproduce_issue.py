import requests
import config
import time

def test_embedding():
    url = f"{config.OLLAMA_BASE_URL}/api/embeddings"
    # Try the newer /api/embed endpoint as well if /api/embeddings is deprecated
    # LangChain might use either
    
    print(f"Testing embedding against {config.OLLAMA_BASE_URL} with model {config.EMBEDDING_MODEL}")
    
    text = "This is a test sentence to check if embeddings are working or returning 500 errors."
    
    payload = {
        "model": config.EMBEDDING_MODEL,
        "prompt": text
    }
    
    try:
        # Try /api/embeddings first (older ollama)
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 404:
            # Try /api/embed (newer ollama)
            print("/api/embeddings not found, trying /api/embed")
            url = f"{config.OLLAMA_BASE_URL}/api/embed"
            payload = {
                "model": config.EMBEDDING_MODEL,
                "input": text
            }
            response = requests.post(url, json=payload, timeout=30)

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Embedding generated.")
        else:
            print(f"Failed! Response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_embedding()
