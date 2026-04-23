from pathlib import Path
from uuid import uuid4


class TempFileManager:
    def __init__(self, base_dir: str = "temp_uploads"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> str:
        safe_name = Path(filename).name
        unique_name = f"{uuid4().hex}_{safe_name}"
        output_path = self.base_dir / unique_name
        output_path.write_bytes(content)
        return str(output_path)