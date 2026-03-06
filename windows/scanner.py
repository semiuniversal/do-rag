import os
import json
import sys
import time

def scan(directories, extensions, exclusions):
    files = []
    
    # Pre-process exclusions for faster lookup
    exclude_names = set(exclusions)
    extensions_set = set(ext.lower() for ext in extensions)

    def _scan_recursive(path):
        try:
            with os.scandir(path) as it:
                entries = list(it)

                subdirs = []
                
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    
                    if entry.is_dir():
                        if entry.name not in exclude_names:
                            subdirs.append(entry.path)
                        continue

                    # File processing logic mirrors indexer.py
                    if (entry.name.endswith(".log") or 
                        entry.name.endswith(".log.txt") or 
                        "_logs" in entry.name or 
                        entry.name.startswith("log-")):
                        continue

                    _, ext = os.path.splitext(entry.name)
                    if ext.lower() not in extensions_set:
                         continue

                    try:
                        stat = entry.stat()
                        if stat.st_size > 5 * 1024 * 1024:
                             continue
                             
                        files.append({
                            "path": entry.path,
                            "mtime": stat.st_mtime,
                            "size": stat.st_size
                        })
                        
                    except OSError:
                        pass
                
                for subdir in subdirs:
                     _scan_recursive(subdir)

        except OSError as e:
            # We output errors to stderr so they don't corrupt the JSON on stdout
            sys.stderr.write(f"Error scanning {path}: {e}\n")

    for directory in directories:
        if os.path.exists(directory):
            _scan_recursive(directory)
        else:
             sys.stderr.write(f"Directory not found: {directory}\n")

    return files

if __name__ == "__main__":
    # Expect JSON config on stdin
    try:
        config = json.load(sys.stdin)
        directories = config.get("directories", [])
        extensions = config.get("extensions", [])
        exclusions = config.get("exclusions", [])
        
        start_time = time.time()
        results = scan(directories, extensions, exclusions)
        end_time = time.time()
        
        output = {
            "files": results,
            "duration": end_time - start_time,
            "count": len(results)
        }
        
        # Print JSON to stdout
        print(json.dumps(output))
        
    except Exception as e:
        sys.stderr.write(f"Fatal error in scanner: {e}\n")
        sys.exit(1)
