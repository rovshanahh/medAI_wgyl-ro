from pathlib import Path


class IngestionValidator:
    SUPPORTED_EXTENSIONS = {".dcm", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    DICOM_SIZE_LIMIT_BYTES = 500 * 1024 * 1024
    NON_DICOM_SIZE_LIMIT_BYTES = 50 * 1024 * 1024

    def validate(self, filename: str, content_type: str | None, content: bytes) -> dict:
        extension = Path(filename).suffix.lower()

        self._validate_format(extension)
        self._validate_size(extension, content)
        self._validate_header_integrity(extension, content)
        self._validate_metadata_readability(filename, content_type)
        self._validate_pixel_array_accessibility(content)
        self._validate_modality_consistency(extension, content_type)

        return {
            "valid": True,
            "extension": extension,
            "content_type": content_type,
            "size_bytes": len(content),
        }

    def _validate_format(self, extension: str) -> None:
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats are DICOM, JPEG, PNG, and TIFF."
            )

    def _validate_size(self, extension: str, content: bytes) -> None:
        size_bytes = len(content)

        if extension == ".dcm":
            if size_bytes > self.DICOM_SIZE_LIMIT_BYTES:
                raise ValueError("DICOM upload exceeds 500 MB limit")
            return

        if size_bytes > self.NON_DICOM_SIZE_LIMIT_BYTES:
            raise ValueError("Non-DICOM upload exceeds 50 MB limit")

    def _validate_header_integrity(self, extension: str, content: bytes) -> None:
        if not content:
            raise ValueError("Uploaded file is empty")

        if extension in {".jpg", ".jpeg"}:
            if len(content) < 4 or content[:2] != b"\xff\xd8":
                raise ValueError("Invalid JPEG header")

        elif extension == ".png":
            if len(content) < 8 or content[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Invalid PNG header")

        elif extension in {".tif", ".tiff"}:
            if len(content) < 4 or content[:4] not in {
                b"II*\x00",
                b"MM\x00*",
            }:
                raise ValueError("Invalid TIFF header")

        elif extension == ".dcm":
            if len(content) < 132:
                raise ValueError("Invalid DICOM file: too small")
            # permissive MVP check
            if content[128:132] != b"DICM":
                raise ValueError("Invalid DICOM header")

    def _validate_metadata_readability(self, filename: str, content_type: str | None) -> None:
        if not filename or not filename.strip():
            raise ValueError("Filename is missing")

        if content_type is not None and not isinstance(content_type, str):
            raise ValueError("Unreadable content type metadata")

    def _validate_pixel_array_accessibility(self, content: bytes) -> None:
        if len(content) < 32:
            raise ValueError("Uploaded file is too small to contain valid image data")

    def _validate_modality_consistency(self, extension: str, content_type: str | None) -> None:
        if content_type is None:
            return

        normalized = content_type.lower()

        if extension in {".jpg", ".jpeg"} and "jpeg" not in normalized and "jpg" not in normalized:
            raise ValueError("JPEG extension does not match content type")

        if extension == ".png" and "png" not in normalized:
            raise ValueError("PNG extension does not match content type")

        if extension in {".tif", ".tiff"} and "tiff" not in normalized and "tif" not in normalized:
            raise ValueError("TIFF extension does not match content type")

        if extension == ".dcm":
            dicom_like = (
                "dicom" in normalized
                or "application/octet-stream" in normalized
                or "application/dicom" in normalized
            )
            if not dicom_like:
                raise ValueError("DICOM extension does not match content type")