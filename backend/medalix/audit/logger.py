import json
from datetime import datetime, UTC
from pathlib import Path


class Logger:
    def __init__(self, log_path: str = "logs/audit_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, event: dict) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            **event,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def info(self, event: dict) -> None:
        self._write("INFO", event)

    def warn(self, event: dict) -> None:
        self._write("WARN", event)

    def error(self, event: dict) -> None:
        self._write("ERROR", event)