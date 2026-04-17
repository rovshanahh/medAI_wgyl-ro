import hashlib
from pathlib import Path


class ChecksumValidator:
    @staticmethod
    def sha256_for_file(file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        sha = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def validate(self, file_path: str, expected_checksum: str) -> bool:
        actual = self.sha256_for_file(file_path)
        return actual == expected_checksum