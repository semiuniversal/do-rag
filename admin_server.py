import os
import json
import logging
import threading
import asyncio
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import config
import indexer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Indexer Integration ---
indexer_thread = None

def run_indexer_job():
    """Runs the indexing job in a separate thread with its own event loop."""
    job = indexer.get_current_job()
    try:
        asyncio.run(job.start())
    except Exception as e:
        logging.error(f"Indexer thread failed: {e}")

def get_dirty_status():
    """Check if settings have changed since last index."""
    try:
        settings_path = Path("settings.json")
        state_path = Path("indexing_state.json")
        
        if not settings_path.exists():
            return False
            
        if not state_path.exists():
            # If we have settings but no state, we definitely need to index
            return True
            
        # If settings were modified AFTER the last index state save
        if settings_path.stat().st_mtime > state_path.stat().st_mtime:
            return True
            
        return False
    except Exception:
        return False

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html", webui_url=config.WEBUI_URL)

@app.route("/config")
def config_page():
    return render_template("config.html")

# --- API ---

@app.route("/api/status")
def api_status():
    # Read indexing state
    state_file = Path("indexing_state.json")
    total_files = 0
    indexed_files = []
    
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
                        "chunks": len(data.get("chunk_ids", []))
                    })
        except Exception as e:
            logging.error(f"Error reading state: {e}")

    # Sort by recent
    indexed_files.sort(key=lambda x: x["mtime"] or 0, reverse=True)

    return jsonify({
        "total_files": total_files,
        "indexed_files": indexed_files,
        "indexer_running": False # TODO: check real status if possible
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

@app.route("/api/indexer/status")
def api_indexer_status():
    job = indexer.get_current_job()
    status = job.get_status()
    # Enhance with dirty flag if needed
    status["dirty"] = get_dirty_status()
    return jsonify(status)

@app.route("/api/indexer/start", methods=["POST"])
def api_indexer_start():
    global indexer_thread
    job = indexer.get_current_job()
    
    if job.status in ("preparing", "scanning", "indexing"):
        return jsonify({"error": "Indexing already in progress"}), 400
        
    # Start in background thread
    indexer_thread = threading.Thread(target=run_indexer_job, daemon=True)
    indexer_thread.start()
    
    return jsonify({"status": "started"})

@app.route("/api/indexer/stop", methods=["POST"])
def api_indexer_stop():
    job = indexer.get_current_job()
    job.cancel()
    return jsonify({"status": "stopping"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
