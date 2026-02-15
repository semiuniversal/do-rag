import json
import os
from pathlib import Path
import config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

STATE_FILE = Path(__file__).parent / "indexing_state.json"

def get_params():
    return {
        "directories": config.DOCUMENT_DIRECTORIES,
        "extensions": config.SUPPORTED_EXTENSIONS,
        "exclusions": config.IGNORED_DIRECTORIES
    }

def scan_files(directories, extensions, exclusions):
    """Recursively find all files, classifying them as indexed_candidate or skipped."""
    candidates = []
    skipped = []
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            logging.warning(f"Directory not found: {directory}")
            continue

        for root, dirs, filenames in os.walk(path):
            # Prune excluded dirs
            dirs[:] = [d for d in dirs if d not in exclusions and not d.startswith(".")]
            
            if any(part.startswith(".") for part in Path(root).parts):
                continue

            for filename in filenames:
                if filename.startswith("."):
                    continue
                    
                file_path = Path(root) / filename
                if file_path.suffix.lower() in extensions:
                    candidates.append(str(file_path))
                else:
                    skipped.append((str(file_path), "Unsupported extension"))
                    
    return candidates, skipped

def main():
    print(f"--- Index Transparency Report ---")
    
    # 1. Load Index State
    if not STATE_FILE.exists():
        print("❌ No index state file found. Run indexer first.")
        return
        
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except Exception as e:
        print(f"❌ Error reading state file: {e}")
        return

    indexed_files = set(state.keys())
    print(f"✅ Indexed Files: {len(indexed_files)}")
    
    # 2. Scan filesystem to find what *should* be there vs what is
    params = get_params()
    print(f"📂 Scanning directories: {params['directories']}")
    
    candidates, fs_skipped = scan_files(params["directories"], params["extensions"], params["exclusions"])
    candidate_set = set(candidates)
    
    # 3. Compare
    missing_from_index = candidate_set - indexed_files
    unexpected_in_index = indexed_files - candidate_set # maybe deleted files?
    
    # 4. Report
    print("\n--- Files in Index ---")
    for f in sorted(indexed_files):
        print(f"  [OK] {f}")

    if missing_from_index:
        print("\n--- ⚠️ Files Detectable but NOT Indexed (Missing) ---")
        for f in sorted(missing_from_index):
            print(f"  [MISSING] {f}")
    else:
        print("\n🎉 All transparently detectable files are indexed.")

    if unexpected_in_index:
        print("\n--- 🗑️ Files in Index but not on Disk (Stale) ---")
        for f in sorted(unexpected_in_index):
            print(f"  [STALE] {f}")

if __name__ == "__main__":
    main()
