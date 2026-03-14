import logging
import sys
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a structured logger for the given module name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_StructuredFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


class _StructuredFormatter(logging.Formatter):
    """Emits log records as key=value pairs for easy parsing."""

    LEVEL_LABELS = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        parts: list[str] = [
            f"ts={self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}",
            f"level={self.LEVEL_LABELS.get(record.levelno, record.levelname)}",
            f"logger={record.name}",
            f"msg={record.getMessage()!r}",
        ]

        # Attach any extra fields the caller passed via the extra= kwarg
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip:
                parts.append(f"{key}={value!r}")

        if record.exc_info:
            parts.append(f"exc={self.formatException(record.exc_info)!r}")

        return " ".join(parts)


def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    """Convenience wrapper for emitting a named event with structured fields."""
    logger.info(event, extra=kwargs)
