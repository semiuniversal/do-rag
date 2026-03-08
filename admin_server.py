import os
import json
import logging
import threading
import asyncio
import subprocess
from pathlib import Path
from typing import Optional
from flask import Flask, render_template, request, jsonify, send_from_directory
import config
import logging_config
import settings

logging_config.setup_logging()

import indexer

app = Flask(__name__)

# --- Indexer Integration ---
indexer_thread = None

def run_indexer_job(reset: bool = False, errors_only: bool = False):
    """Runs the indexing job in a separate thread with its own event loop."""
    job = indexer.get_current_job()
    try:
        asyncio.run(job.start(reset=reset, errors_only=errors_only))
    except Exception as e:
        logging.error(f"Indexer thread failed: {e}")

def get_dirty_status():
    """True if indexing-relevant config (directories, extensions, exclusions) changed since last index."""
    try:
        snapshot = indexer._load_index_config_snapshot()
        if not snapshot:
            return False  # No prior index; nothing to compare

        current = settings.load_settings()
        cur_dirs = sorted(current.get("directories") or current.get("DOCUMENT_DIRECTORIES") or [])
        cur_exts = sorted(current.get("extensions") or current.get("SUPPORTED_EXTENSIONS") or [])
        cur_excl = sorted(current.get("exclusions") or current.get("IGNORED_DIRECTORIES") or [])

        snap_dirs = sorted(snapshot.get("directories", []))
        snap_exts = sorted(snapshot.get("extensions", []))
        snap_excl = sorted(snapshot.get("exclusions", []))

        return cur_dirs != snap_dirs or cur_exts != snap_exts or cur_excl != snap_excl
    except Exception:
        return False

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html", webui_url=config.WEBUI_URL, current_page="dashboard")

@app.route("/config")
def config_page():
    return render_template("config.html", current_page="config")


@app.route("/logs")
def logs_page():
    return render_template("logs.html", current_page="logs")

# --- API ---

def _check_webui_healthy() -> bool:
    """Check if Open WebUI is responding to health endpoint."""
    try:
        import urllib.request
        url = f"{config.WEBUI_URL.rstrip('/')}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


@app.route("/api/status")
def api_status():
    # Read indexing state
    state_file = Path("indexing_state.json")
    total_files = 0
    indexed_files = []

    # Load persisted error files (survives restarts)
    error_paths = set(indexer._load_error_files())

    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                total_files = len(state)
                # Convert to list for frontend
                for path, data in state.items():
                    indexed_files.append({
                        "path": path,
                        "mtime": data.get("mtime"),
                        "chunks": len(data.get("chunk_ids", [])),
                        "error": path in error_paths,
                    })
        except Exception as e:
            logging.error(f"Error reading state: {e}")

    # Sort by recent
    indexed_files.sort(key=lambda x: x["mtime"] or 0, reverse=True)

    return jsonify({
        "total_files": total_files,
        "indexed_files": indexed_files,
        "indexer_running": indexer.get_current_job().status in ("preparing", "scanning", "indexing"),
        "webui_healthy": _check_webui_healthy(),
    })

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        new_settings = request.json
        # Basic validation could go here
        if config.save_settings(new_settings):
            # Reload config in memory? 
            # In a real app we might need to signal the other processes to reload
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Failed to save"}), 500
    
    # GET
    settings = config.load_settings()
    return jsonify(settings)

from functools import lru_cache

@lru_cache(maxsize=100)
def get_directory_listing(path_str):
    target = Path(path_str)
    if not target.exists() or not target.is_dir():
        return None

    items = []
    try:
        # List directories first, then files
        for item in target.iterdir():
            try:
                if item.name.startswith("."): # Skip hidden
                    continue
                
                entry = {
                    "name": item.name,
                    "path": str(item.absolute()),
                    "is_dir": item.is_dir()
                }
                items.append(entry)
            except (PermissionError, OSError) as e:
                logging.warning(f"Skipping inaccessible file {item}: {e}")
                continue
            
        # Sort: Directories first, then alphabetical
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        return items
    except Exception as e:
        return str(e)

@app.route("/api/browse")
def api_browse():
    path = request.args.get("path", "/")
    
    # Security: Prevent escaping root if desired, but this is a local tool
    result = get_directory_listing(path)
    
    if result is None:
        return jsonify({"error": "Invalid directory"}), 400
    if isinstance(result, str):
         return jsonify({"error": result}), 500

    target = Path(path) # Keep original target for existence check or use absolute?
    # Better to resolve once
    try:
        abs_target = target.resolve()
    except Exception:
        # Fallback if resolve fails (e.g. permissions?), though exists() check usually catches this
        abs_target = target.absolute()

    return jsonify({
        "current": str(abs_target),
        "parent": str(abs_target.parent),
        "items": result
    })

@app.route("/api/models")
def api_models():
    try:
        # interacting with ollama CLI is often more reliable for 'list' than the client if not configured perfectly
        # but let's try the simple HTTP approach since we know the URL
        import requests
        url = f"{config.OLLAMA_BASE_URL}/api/tags"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return jsonify({"models": models})
        else:
            return jsonify({"error": "Failed to fetch models from Ollama"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/webui/restart", methods=["POST"])
def api_webui_restart():
    """Restart Open WebUI container only (for recovery when unhealthy)."""
    project_root = Path(__file__).parent.resolve()
    run_webui = project_root / "run_webui.sh"
    if not run_webui.exists():
        return jsonify({"error": "run_webui.sh not found"}), 404
    try:
        subprocess.Popen(
            ["/bin/bash", str(run_webui)],
            cwd=str(project_root),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return jsonify({"status": "restart_initiated", "message": "Open WebUI restart started. The button will enable when it's ready (usually 1–2 minutes)."})
    except Exception as e:
        logging.error(f"Failed to start WebUI restart: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/restart", methods=["POST"])
def api_restart():
    """Run run.sh to restart all services (Ollama, Qdrant, MCP, Admin Portal, WebUI)."""
    project_root = Path(__file__).parent.resolve()
    run_script = project_root / "run.sh"
    if not run_script.exists():
        return jsonify({"error": "run.sh not found"}), 404
    try:
        subprocess.Popen(
            ["/bin/bash", str(run_script)],
            cwd=str(project_root),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return jsonify({"status": "restart_initiated", "message": "Restart started. The application will restart shortly. Please refresh the page in 30–60 seconds."})
    except Exception as e:
        logging.error(f"Failed to start restart: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/pull", methods=["POST"])
def api_pull_model():
    model_name = request.json.get("model")
    if not model_name:
        return jsonify({"error": "Model name required"}), 400
    
    # Spawn a background process to pull
    # utilizing subprocess to call 'ollama pull'
    try:
        # We use Popen without waiting to avoid blocking the server
        # In a production app context, we'd use a task queue (e.g. celery/rq)
        subprocess.Popen(["ollama", "pull", model_name])
        return jsonify({"status": "pull_started", "message": f"Started pulling {model_name}. Check terminal/logs for progress."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Indexer APIs ---

# Log level severity (higher = more severe). Used for "level and above" filtering.
_LEVEL_SEVERITY = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}


def _level_from_line(line: str) -> Optional[str]:
    """Extract log level from a line. Returns None if not parseable."""
    parts = line.split(" | ", 3)
    if len(parts) >= 2:
        level = parts[1].strip().upper()
        if level in _LEVEL_SEVERITY:
            return level
    return None


def _subsystem_from_line(line: str) -> str:
    """Extract or infer subsystem from a log line. Supports old (4-field) and new (5-field) format."""
    parts = line.split(" | ", 4)  # max 5 parts: ts, level, [subsystem], name, message
    if len(parts) >= 5:
        return parts[2].strip()
    if len(parts) >= 4:
        name = parts[2].strip()
        if "werkzeug" in name:
            return "admin-ui"
        if "indexer" in name or "index_docs" in name or "windows" in name:
            return "indexer"
        if "server" in name:
            return "mcp-server"
        if "config" in name or "settings" in name:
            return "config"
    return "system"


def _read_log_lines(log_path: Path, tail: int = None, offset: int = 0, limit: int = 500,
                    level_filter: str = None, search: str = None,
                    subsystem_filter: str = None) -> tuple:
    """Read log lines with optional filters. Returns (lines, next_offset, has_more, total).
    offset: number of lines already loaded from the end (for 'load older').
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()

    if level_filter:
        min_sev = _LEVEL_SEVERITY.get(level_filter.upper(), 0)
        def level_passes(line: str) -> bool:
            lev = _level_from_line(line)
            if lev is None:
                return True  # Include unparseable lines when filtering
            return _LEVEL_SEVERITY.get(lev, 0) >= min_sev
        all_lines = [l for l in all_lines if level_passes(l)]
    if subsystem_filter:
        all_lines = [l for l in all_lines if _subsystem_from_line(l) == subsystem_filter]
    if search:
        search_lower = search.lower()
        all_lines = [l for l in all_lines if search_lower in l.lower()]

    total = len(all_lines)
    if tail:
        lines = all_lines[-min(tail, len(all_lines)):]
        next_offset = 0
        has_more = False
    else:
        # offset=0: last 500. offset=500: next 500 older.
        n = len(all_lines)
        start = max(0, n - offset - limit)
        end = n - offset
        lines = all_lines[start:end]
        next_offset = offset + len(lines)
        has_more = start > 0

    return lines, next_offset, has_more, total


@app.route("/api/logs")
def api_logs():
    """Return log lines as JSON with pagination, level filter, and search."""
    log_path = logging_config.get_log_path()
    if not log_path.exists():
        return jsonify({"error": "Log file not yet created", "lines": []}), 404

    try:
        limit = min(int(request.args.get("limit", 500)), 2000)
    except ValueError:
        limit = 500
    tail = request.args.get("tail")
    tail = int(tail) if tail else None
    offset = int(request.args.get("offset", 0))
    level_filter = request.args.get("level") or None
    search = request.args.get("search") or None
    subsystem_filter = request.args.get("subsystem") or None

    lines, next_offset, has_more, total = _read_log_lines(
        log_path, tail=tail, offset=offset, limit=limit,
        level_filter=level_filter, search=search,
        subsystem_filter=subsystem_filter
    )
    lines = [l.rstrip("\n") for l in lines]

    return jsonify({
        "lines": lines,
        "next_offset": next_offset,
        "has_more": has_more,
        "total_lines": total,
        "subsystems": logging_config.get_subsystems(),
    })


@app.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    """Clear the log file. Requires confirmation from the client."""
    if logging_config.clear_log_file():
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Failed to clear logs"}), 500


@app.route("/api/indexer/status")
def api_indexer_status():
    job = indexer.get_current_job()
    status = job.get_status()
    status["dirty"] = get_dirty_status()
    status["log_path"] = str(logging_config.get_log_path())
    return jsonify(status)

@app.route("/api/indexer/start", methods=["POST"])
def api_indexer_start():
    global indexer_thread
    job = indexer.get_current_job()
    
    if job.status in ("preparing", "scanning", "indexing"):
        return jsonify({"error": "Indexing already in progress"}), 400

    reset = False
    errors_only = False
    if request.json:
        reset = request.json.get("reset", False)
        errors_only = request.json.get("errors_only", False)

    indexer_thread = threading.Thread(
        target=run_indexer_job,
        args=(reset, errors_only),
        daemon=True
    )
    indexer_thread.start()

    return jsonify({"status": "started", "reset": reset, "errors_only": errors_only})

@app.route("/api/indexer/stop", methods=["POST"])
def api_indexer_stop():
    job = indexer.get_current_job()
    job.cancel()
    return jsonify({"status": "stopping"})


@app.route("/api/indexer/reindex-file", methods=["POST"])
def api_indexer_reindex_file():
    """Re-index a single file by path."""
    data = request.json or {}
    path = data.get("path") or data.get("file_path")
    if not path or not isinstance(path, str):
        return jsonify({"error": "Missing or invalid 'path'"}), 400
    path = path.strip()
    if not path:
        return jsonify({"error": "Path cannot be empty"}), 400
    success, message = indexer.index_single_file(path)
    if success:
        return jsonify({"status": "ok", "message": message})
    return jsonify({"error": message}), 400


def _wsl_path_to_windows(path: str) -> str:
    """Convert WSL path (/mnt/c/...) to Windows path (C:\\...)."""
    path = path.replace("\\", "/")
    if path.startswith("/mnt/") and len(path) > 5:
        drive = path[5].upper()
        rest = path[6:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return path.replace("/", "\\")


def _is_path_allowed(path: str) -> bool:
    """Ensure path is under a configured document directory (security)."""
    try:
        resolved = Path(path).resolve()
        s = settings.load_settings()
        dirs = s.get("directories") or s.get("DOCUMENT_DIRECTORIES") or []
        for d in dirs:
            try:
                base = Path(d).resolve()
                if str(resolved).startswith(str(base)) or resolved == base:
                    return True
            except (OSError, ValueError):
                continue
        return False
    except (OSError, ValueError):
        return False


@app.route("/api/open-file")
def api_open_file():
    """Open a file in Windows Explorer (WSL interop). Converts /mnt/c/... to C:\\... and runs explorer.exe /select."""
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"error": "Missing path"}), 400
    if not _is_path_allowed(path):
        return jsonify({"error": "Path not under configured directories"}), 403
    if not Path(path).exists() or not Path(path).is_file():
        return jsonify({"error": "File not found"}), 404
    win_path = _wsl_path_to_windows(path)
    try:
        subprocess.Popen(
            ["explorer.exe", f'/select,"{win_path}"'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return jsonify({"status": "ok", "message": "Opening in Explorer"})
    except Exception as e:
        logging.warning(f"Failed to open file: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # use_reloader=False: reloader passes socket FDs to child, which fails under nohup/background
    # (run.sh restarts admin via nohup; reloader causes OSError: [Errno 9] Bad file descriptor)
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
