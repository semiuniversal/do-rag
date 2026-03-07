# logging_config.py
# Centralized logging for do-rag. Call setup_logging() at application startup.

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "do-rag.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(subsystem)-12s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Subsystems for log filtering. Map pathname/module to subsystem tag.
SUBSYSTEMS = ["admin-ui", "indexer", "mcp-server", "config", "containers", "system"]
_SUBSYSTEM_MAP = [
    (["admin_server", "werkzeug"], "admin-ui"),
    (["indexer", "index_docs", "windows"], "indexer"),
    (["server"], "mcp-server"),
    (["config", "settings", "logging_config"], "config"),
    (["run_qdrant", "run_webui", "run_ollama"], "containers"),
]


def _get_max_log_bytes() -> int:
    """Return max log size in bytes (0 = no limit)."""
    try:
        import config
        kb = config.load_settings().get("LOG_MAX_SIZE_KB", 1024)
        return max(0, int(kb)) * 1024
    except Exception:
        return 1024 * 1024  # 1 MB fallback


def _truncate_log_if_needed(log_path: Path, max_bytes: int, handler=None) -> bool:
    """Truncate log file to keep only the last max_bytes, on newline boundaries.
    If handler is provided, closes it before truncating. Returns True if truncation was done."""
    if max_bytes <= 0 or not log_path.exists():
        return False
    try:
        size = log_path.stat().st_size
        if size <= max_bytes:
            return False
        if handler is not None:
            handler.close()
        with open(log_path, "rb") as f:
            f.seek(-max_bytes, 2)
            chunk = f.read(1024)
            idx = chunk.find(b"\n")
            if idx >= 0:
                f.seek(-max_bytes + idx + 1, 2)
            content = f.read()
        with open(log_path, "wb") as f:
            f.write(content)
        return True
    except Exception:
        return False


class TruncatingFileHandler(logging.FileHandler):
    """FileHandler that truncates the log when it exceeds max size."""

    def emit(self, record: logging.LogRecord) -> None:
        max_bytes = _get_max_log_bytes()
        if max_bytes > 0:
            did_truncate = _truncate_log_if_needed(Path(self.baseFilename), max_bytes, handler=self)
            if did_truncate:
                self.stream = self._open()
        super().emit(record)


def _get_subsystem(record: logging.LogRecord) -> str:
    """Derive subsystem from the log record's pathname or logger name."""
    pathname = getattr(record, "pathname", "") or ""
    name = getattr(record, "name", "") or ""
    for patterns, subsystem in _SUBSYSTEM_MAP:
        for p in patterns:
            if p in pathname or p in name:
                return subsystem
    return "system"


class SubsystemFilter(logging.Filter):
    """Add subsystem to each log record for filtering."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.subsystem = _get_subsystem(record)
        return True


_configured = False


def setup_logging(
    log_file: Path = LOG_FILE,
    level: int = logging.INFO,
    also_stderr: bool = None,
) -> Path:
    """Configure root logger to write to a single log file (and optionally stderr).
    Returns the path to the log file.
    When also_stderr is None, stderr is enabled only when stdout is a TTY (interactive).
    """
    global _configured
    if _configured:
        return log_file

    if also_stderr is None:
        also_stderr = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    subsystem_filter = SubsystemFilter()

    # File handler (append, with size-based truncation)
    file_handler = TruncatingFileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    file_handler.addFilter(subsystem_filter)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicates
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(file_handler)

    if also_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(level)
        stderr_handler.addFilter(subsystem_filter)
        root.addHandler(stderr_handler)

    _configured = True
    logging.info(f"Logging to {log_file}")
    return log_file


def get_log_path() -> Path:
    """Return the path to the application log file."""
    return LOG_FILE


def get_subsystems() -> list:
    """Return the list of subsystem names for the log filter dropdown."""
    return list(SUBSYSTEMS)


def clear_log_file() -> bool:
    """Clear the log file. Closes and reopens the file handler. Returns True on success."""
    global _configured
    root = logging.getLogger()
    file_handler = None
    for h in root.handlers[:]:
        if isinstance(h, (logging.FileHandler, TruncatingFileHandler)) and Path(getattr(h, "baseFilename", "")).resolve() == LOG_FILE.resolve():
            file_handler = h
            break
    if file_handler is None:
        return False
    try:
        file_handler.close()
        root.removeHandler(file_handler)
        LOG_FILE.write_text("", encoding="utf-8")
        # Re-add file handler
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        subsystem_filter = SubsystemFilter()
        new_handler = TruncatingFileHandler(LOG_FILE, encoding="utf-8")
        new_handler.setFormatter(formatter)
        new_handler.setLevel(root.level)
        new_handler.addFilter(subsystem_filter)
        root.addHandler(new_handler)
        return True
    except Exception:
        return False
