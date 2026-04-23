from pathlib import Path


class DataRetentionManager:
    def delete_now(self, path: str | None) -> dict:
        if not path:
            return {
                "deleted": False,
                "reason": "No path provided",
                "path": None,
            }

        file_path = Path(path)

        if not file_path.exists():
            return {
                "deleted": False,
                "reason": "File not found",
                "path": str(file_path),
            }

        if file_path.is_dir():
            return {
                "deleted": False,
                "reason": "Expected a file path, got a directory",
                "path": str(file_path),
            }

        file_path.unlink()

        return {
            "deleted": True,
            "reason": "File deleted successfully",
            "path": str(file_path),
        }