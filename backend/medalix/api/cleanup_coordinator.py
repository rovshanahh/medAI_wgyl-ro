class CleanupCoordinator:
    def __init__(self, retention_manager, logger) -> None:
        self._retention_manager = retention_manager
        self._logger = logger

    def cleanup_temp_file(self, temp_path: str | None) -> None:
        if temp_path is None:
            return

        try:
            result = self._retention_manager.delete_now(temp_path)
            self._logger.info({"event": "retention_cleanup", "result": result})
        except Exception as exc:
            self._logger.error(
                {
                    "event": "retention_cleanup_failed",
                    "temp_path": temp_path,
                    "message": str(exc),
                }
            )