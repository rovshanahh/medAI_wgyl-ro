from pathlib import Path
from uuid import uuid4


class TempFileManager:
    def __init__(self, base_dir: str = "temp_uploads"):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> str:
        if not content:
            raise ValueError("Uploaded file content is empty")

        safe_name = Path(filename or "upload").name
        safe_name = safe_name.replace("/", "_").replace("\\", "_").strip()

        if not safe_name:
            safe_name = "upload"

        unique_name = f"{uuid4().hex}_{safe_name}"
        output_path = (self.base_dir / unique_name).resolve()

        if self.base_dir not in output_path.parents:
            raise ValueError("Resolved upload path escaped temp upload directory")

        output_path.write_bytes(content)
        return str(output_path)