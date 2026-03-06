
import logging
import sys
import time
import os
from pathlib import Path

# Setup logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

print(f"Current Working Directory: {os.getcwd()}")
print(f"User: {os.environ.get('USER')}")

try:
    import settings
    import indexer
    import windows.bridge as bridge
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def debug_scan():
    print("\n--- DATA DUMP ---")
    s = settings.load_settings()
    print(f"Settings loaded: optimize_for_windows={s.get('optimize_for_windows')}")
    
    directories = s.get("directories", [])
    print(f"Directories to scan: {directories}")
    
    if not directories:
        print("No directories configured! This might be why it's fast/slow/broken.")
    
    print("\n--- BRIDGE CHECK ---")
    is_wsl = bridge.is_wsl()
    print(f"is_wsl(): {is_wsl}")
    
    if directories:
        win_path = bridge.to_windows_path(directories[0])
        print(f"Sample path conversion: {directories[0]} -> {win_path}")

    print("\n--- RUNNING SCAN ---")
    start = time.time()
    files = indexer._scan_files(directories, s.get("extensions", []), s.get("exclusions", []))
    end = time.time()
    
    print(f"\nScan complete in {end - start:.4f} seconds.")
    print(f"Found {len(files)} files.")
    
    # Check if we can identify if bridge was used from the output or side effects?
    # The scan function logs "Attempting Windows Native Scan..." if it tries.

if __name__ == "__main__":
    debug_scan()
