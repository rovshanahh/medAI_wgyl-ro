from pathlib import Path


class TempFileManager:
    def __init__(self, base_dir: str = "temp_uploads"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> str:
        safe_name = Path(filename).name
        output_path = self.base_dir / safe_name
        output_path.write_bytes(content)
        return str(output_path)