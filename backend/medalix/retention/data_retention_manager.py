from pathlib import Path


class DataRetentionManager:
    def __init__(self, allowed_base_dir: str = "temp_uploads"):
        self.allowed_base_dir = Path(allowed_base_dir).resolve()
        self.allowed_base_dir.mkdir(parents=True, exist_ok=True)

    def delete_now(self, path: str | None) -> dict:
        if not path:
            return {
                "deleted": False,
                "reason": "No path provided",
                "path": None,
            }

        file_path = Path(path).resolve()

        if self.allowed_base_dir not in file_path.parents:
            return {
                "deleted": False,
                "reason": "Refused to delete file outside temp upload directory",
                "path": str(file_path),
            }

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