class IngestionValidator:
    ALLOWED_TYPES = {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "application/dicom",
        "application/octet-stream",
    }

    MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

    def validate(self, filename: str, content_type: str, content: bytes) -> None:
        if content_type not in self.ALLOWED_TYPES:
            raise ValueError("Unsupported file type")

        if not content:
            raise ValueError("Empty file")

        if len(content) > self.MAX_SIZE_BYTES:
            raise ValueError("File too large")