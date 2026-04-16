from pathlib import Path


class DataRetentionManager:
    def delete_now(self, path: str) -> dict:
        file_path = Path(path)

        if not file_path.exists():
            return {
                "deleted": False,
                "reason": "File not found",
                "path": str(file_path),
            }

        file_path.unlink()

        return {
            "deleted": True,
            "reason": "File deleted successfully",
            "path": str(file_path),
        }