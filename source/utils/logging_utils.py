"""Structured logging and JSONL telemetry logging utilities.

Fulfills Constitution C38.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any


def setup_logger(name: str, log_dir: Optional[str] = None, level: str = 'INFO') -> logging.Logger:
    """Create and configure a Logger instance with console and optional file handler.

    Args:
        name: Name of the logger.
        log_dir: Optional directory path to write {name}.log file.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler if log_dir provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_to_jsonl(filepath: str, entry: Dict[str, Any]) -> None:
    """Append a structured dictionary entry as a single JSON line to a JSONL file.

    Adds an automatic ISO 8601 UTC timestamp field if not present.

    Args:
        filepath: Target .jsonl file path.
        entry: Dictionary of log data.
    """
    log_data = entry.copy()
    if 'timestamp' not in log_data:
        log_data['timestamp'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_data) + '\n')
        f.flush()
