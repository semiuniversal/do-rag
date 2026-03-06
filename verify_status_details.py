
import requests
import time
import threading

def trigger_index():
    try:
        print("Triggering index...")
        requests.post("http://localhost:5001/api/indexer/start")
    except Exception as e:
        print(f"Trigger failed: {e}")

def monitor_status():
    print("Monitoring status for 10 seconds...")
    seen_details = set()
    for _ in range(20):
        try:
            resp = requests.get("http://localhost:5001/api/indexer/status")
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                detail = data.get("detail_status")
                print(f"Status: {status} | Detail: {detail}")
                
                if detail:
                    seen_details.add(detail)
                    
                if status == "indexing":
                    print("Reached indexing phase. Stopping monitor.")
                    break
        except Exception as e:
            print(f"Monitor error: {e}")
        time.sleep(0.5)
    
    print("\n--- Unique Details Seen ---")
    for d in seen_details:
        print(f"- {d}")

if __name__ == "__main__":
    # Start index in background (it blocks if called directly? No, API is async spawn)
    trigger_index()
    monitor_status()
