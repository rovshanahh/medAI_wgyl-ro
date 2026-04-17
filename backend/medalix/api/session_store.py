import time
import uuid


class SessionStore:
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self._results: dict[str, dict] = {}
        self._created_at: dict[str, float] = {}
        self._ttl_seconds = ttl_seconds

    def create_analysis_id(self) -> str:
        self.evict_expired()
        return str(uuid.uuid4())

    def save_result(self, analysis_id: str, result: dict) -> None:
        self.evict_expired()
        self._results[analysis_id] = result
        self._created_at[analysis_id] = time.time()

    def get_result(self, analysis_id: str) -> dict | None:
        self.evict_expired()

        created_at = self._created_at.get(analysis_id)
        if created_at is None:
            return None

        if time.time() - created_at > self._ttl_seconds:
            self.delete_result(analysis_id)
            return None

        return self._results.get(analysis_id)

    def delete_result(self, analysis_id: str) -> None:
        self._results.pop(analysis_id, None)
        self._created_at.pop(analysis_id, None)

    def evict_expired(self) -> None:
        now = time.time()
        expired_ids = [
            analysis_id
            for analysis_id, created_at in self._created_at.items()
            if now - created_at > self._ttl_seconds
        ]
        for analysis_id in expired_ids:
            self.delete_result(analysis_id)