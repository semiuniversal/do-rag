
import os
import time
from pathlib import Path

def debug_dirty_status():
    settings_path = Path("settings.json")
    state_path = Path("indexing_state.json")
    
    print(f"--- Debugging Dirty Status ---")
    
    if not settings_path.exists():
        print(f"settings.json: MISSING")
    else:
        mtime = settings_path.stat().st_mtime
        print(f"settings.json: {settings_path.resolve()} | mtime: {mtime} ({time.ctime(mtime)})")

    if not state_path.exists():
        print(f"indexing_state.json: MISSING")
    else:
        mtime = state_path.stat().st_mtime
        print(f"indexing_state.json: {state_path.resolve()} | mtime: {mtime} ({time.ctime(mtime)})")

    if settings_path.exists() and state_path.exists():
        diff = settings_path.stat().st_mtime - state_path.stat().st_mtime
        print(f"Difference (settings - state): {diff} seconds")
        if diff > 0:
            print("RESULT: DIRTY (settings is newer)")
        else:
            print("RESULT: CLEAN (state is newer or equal)")

if __name__ == "__main__":
    debug_dirty_status()
