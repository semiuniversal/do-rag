import os
import json
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import config
from indexer import get_current_job

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

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

@app.route("/api/browse")
def api_browse():
    path = request.args.get("path", "/")
    
    # Security: Prevent escaping root if desired, but this is a local tool
    target = Path(path)
    if not target.exists() or not target.is_dir():
        return jsonify({"error": "Invalid directory"}), 400

    items = []
    try:
        # List directories first, then files
        for item in target.iterdir():
            if item.name.startswith("."): # Skip hidden
                continue
            
            entry = {
                "name": item.name,
                "path": str(item.absolute()),
                "is_dir": item.is_dir()
            }
            items.append(entry)
            
        # Sort: Directories first, then alphabetical
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "current": str(target.absolute()),
        "parent": str(target.parent.absolute()),
        "items": items
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
