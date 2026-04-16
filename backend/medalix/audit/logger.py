import json
from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, log_path: str = "logs/audit_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, event: dict) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "INFO",
            **event,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")