import requests
import json
import sys

BASE_URL = "http://localhost:5001"

def test_navigation():
    print("Testing Navigation Logic...")
    
    # 1. Start at current directory (".")
    print("\n1. Loading start path ('.')...")
    resp = requests.get(f"{BASE_URL}/api/browse?path=.")
    if resp.status_code != 200:
        print("FAILED: Could not load start path")
        return
    
    data = resp.json()
    start_current = data['current']
    start_parent = data['parent']
    print(f"   Current: {start_current}")
    print(f"   Parent:  {start_parent}")
    
    if start_current == start_parent:
        print("FAILED: Current == Parent at start (Root stuck issue?)")
        return

    # 2. Simulate "Click Up"
    print(f"\n2. Simulating 'Up' Click (navigating to {start_parent})...")
    resp = requests.get(f"{BASE_URL}/api/browse", params={"path": start_parent})
    
    if resp.status_code != 200:
        print(f"FAILED: Could not load parent path: {resp.text}")
        return

    data = resp.json()
    new_current = data['current']
    new_parent = data['parent']
    print(f"   New Current: {new_current}")
    print(f"   New Parent:  {new_parent}")

    # Verification
    if new_current == start_parent:
        print("\n[SUCCESS] Navigation confirmed. 'Up' button logic is valid.")
    else:
        print(f"\n[FAILED] Expected current to be {start_parent}, got {new_current}")

if __name__ == "__main__":
    try:
        test_navigation()
    except Exception as e:
        print(f"Error: {e}")
