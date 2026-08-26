import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any
import uuid

import os
LOG_JSON = os.getenv("LOG_JSON", "false").lower() == "true"

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, default=str)

def setup_logging():
    """Configure logging with console and rotating file handlers."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "app.log")

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "level": log_level,
        }
    }
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": 10_485_760,  # 10 MB
            "backupCount": 5,
            "level": log_level,
        }

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter,
            },
            "plain": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": list(handlers.keys()),
        },
    }

    # Choose formatter
    formatter_name = "json" if LOG_JSON else "plain"
    for handler in logging_config["handlers"].values():
        handler["formatter"] = formatter_name

    logging.config.dictConfig(logging_config)

# Get a logger instance for a given module
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)