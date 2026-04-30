from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_LOGGER_NAME = "rtbench"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_RUN_LOG_FILENAME = "run.jsonl"
DEFAULT_CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DEFAULT_CONSOLE_DATEFMT = "%Y-%m-%d %H:%M:%S"
_RESERVED_LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    text = str(level or DEFAULT_LOG_LEVEL).strip().upper()
    return int(getattr(logging, text, logging.INFO))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = {
            key: _json_safe(value)
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_FIELDS and not key.startswith("_")
        }
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _build_console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_CONSOLE_FORMAT, datefmt=DEFAULT_CONSOLE_DATEFMT))
    return handler


def _build_json_handler(path: str | Path, level: int) -> logging.Handler:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(JsonLinesFormatter())
    return handler


def default_run_log_path(run_dir: str | Path, *, filename: str = DEFAULT_RUN_LOG_FILENAME) -> Path:
    return Path(run_dir) / "logs" / filename


def configure_logging(
    *,
    level: str | int | None = None,
    logger_name: str = DEFAULT_LOGGER_NAME,
    json_log_path: str | Path | None = None,
    console: bool = True,
) -> logging.Logger:
    resolved_level = _coerce_level(level)
    logger = logging.getLogger(logger_name)
    logger.setLevel(resolved_level)
    logger.propagate = False
    _close_handlers(logger)
    if console:
        logger.addHandler(_build_console_handler(resolved_level))
    if json_log_path is not None:
        logger.addHandler(_build_json_handler(json_log_path, resolved_level))
    return logger


@contextmanager
def attach_json_log(
    json_log_path: str | Path,
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    level: str | int | None = None,
) -> Iterator[Path]:
    logger = logging.getLogger(logger_name)
    handler = _build_json_handler(json_log_path, _coerce_level(level or logger.level or DEFAULT_LOG_LEVEL))
    logger.addHandler(handler)
    try:
        yield Path(json_log_path)
    finally:
        logger.removeHandler(handler)
        handler.close()
