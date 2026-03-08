
import os
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

def is_wsl() -> bool:
    """Check if running in WSL."""
    if "WSL_DISTRO_NAME" in os.environ:
        return True
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                return True
    except:
        pass
    return False

def to_windows_path(linux_path: str) -> str:
    """Convert a WSL path to a Windows path using wslpath."""
    try:
        result = subprocess.run(["wslpath", "-w", linux_path], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.warning(f"wslpath failed for {linux_path}: {e}")
        return linux_path # Fallback

def to_linux_path(windows_path: str) -> str:
    """Convert a Windows path back to WSL path."""
    try:
        result = subprocess.run(["wslpath", "-u", windows_path], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.warning(f"wslpath failed for {windows_path}: {e}")
        return windows_path

def scan_windows(directories: List[str], extensions: List[str], exclusions: List[str]) -> Optional[List[Tuple[Path, float]]]:
    """
    Invoke the Windows-side scanner script.
    Returns None if the bridge fails or is not applicable.
    """
    if not is_wsl():
        return None

    # Only use bridge if ALL directories are in /mnt/c (or other mounts)
    # Actually, simpler: just try to convert paths. If conversion fails, we skip bridge.
    win_dirs = []
    for d in directories:
        if not d.startswith("/mnt/"): # heuristic
             return None
        wd = to_windows_path(d)
        if wd == d: # Conversion failed or no change
            return None
        win_dirs.append(wd)
        
    config = {
        "directories": win_dirs,
        "extensions": extensions,
        "exclusions": exclusions
    }
    
    scanner_script = Path(__file__).parent / "scanner.py"
    if not scanner_script.exists():
        logging.error("Windows scanner script not found.")
        return None
        
    # Convert script path to windows format for python.exe
    win_script_path = to_windows_path(str(scanner_script))

    # UNC paths (\\wsl.localhost\...) cause "UNC paths are not supported" in Windows Python.
    # Run from WSL instead when script would be UNC.
    use_wsl = win_script_path.startswith("\\\\")

    try:
        if use_wsl:
            # Run scanner in WSL with Linux paths - avoids UNC path issues
            cmd = ["python3", str(scanner_script)]
            config = {"directories": directories, "extensions": extensions, "exclusions": exclusions}
            logging.info("Invoking scanner via WSL (project in WSL home; UNC paths not supported by Windows Python)")
        else:
            # Run Windows Python with drive-letter path
            cmd = ["cmd.exe", "/c", "python", win_script_path]
            config = {"directories": win_dirs, "extensions": extensions, "exclusions": exclusions}
            logging.info(f"Invoking Windows scanner: {cmd}")
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=json.dumps(config))

        if process.returncode != 0:
            logging.error(f"Scanner failed (code {process.returncode}): {stderr}")
            return None

        if stderr and "UNC paths" not in stderr:
            logging.warning(f"Scanner stderr: {stderr}")

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse scanner output: {e}. Output start: {stdout[:100]}")
            return None

        files = []
        if use_wsl:
            # Paths are already Linux format
            for item in result.get("files", []):
                files.append((Path(item["path"]), item["mtime"]))
        else:
            # Convert Windows paths back to Linux
            path_mappings = []
            for i, win_dir in enumerate(win_dirs):
                linux_dir = directories[i]
                w_p = win_dir.rstrip("\\")
                l_p = linux_dir.rstrip("/")
                path_mappings.append((w_p, l_p))

            for item in result.get("files", []):
                win_path = item["path"]
                mtime = item["mtime"]
                linux_path_str = None
                for w_prefix, l_prefix in path_mappings:
                    if win_path.startswith(w_prefix):
                        suffix = win_path[len(w_prefix):].replace("\\", "/")
                        linux_path_str = l_prefix + suffix
                        break
                if not linux_path_str:
                    linux_path_str = to_linux_path(win_path)
                files.append((Path(linux_path_str), mtime))
            
        logging.info(f"Windows scanner returned {len(files)} files in {result.get('duration', 0):.2f}s")
        return files

    except Exception as e:
        logging.error(f"Bridge execution failed: {e}")
        return None
