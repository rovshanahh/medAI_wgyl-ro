from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"


class IngestionValidator:
    SUPPORTED_EXTENSIONS = {".dcm", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    DICOM_SIZE_LIMIT_BYTES = 500 * 1024 * 1024
    NON_DICOM_SIZE_LIMIT_BYTES = 50 * 1024 * 1024

    def validate(self, filename: str, content_type: str | None, content: bytes) -> dict:
        self._validate_filename(filename)

        extension = Path(filename).suffix.lower()

        self._validate_format(extension)
        self._validate_size(extension, content)
        self._validate_header_integrity(extension, content)
        self._validate_content_type(extension, content_type)

        # Important:
        # DICOM is not readable by PIL. It is handled later by DicomConverter.
        if extension == ".dcm":
            return {
                "valid": True,
                "extension": extension,
                "content_type": content_type,
                "size_bytes": len(content),
                "image_width": None,
                "image_height": None,
                "image_mode": "DICOM",
                "requires_conversion": True,
                "message": "DICOM header validated. Pixel extraction will be handled by DicomConverter.",
            }

        image_info = self._validate_image_readability(content)

        return {
            "valid": True,
            "extension": extension,
            "content_type": content_type,
            "size_bytes": len(content),
            "image_width": image_info["width"],
            "image_height": image_info["height"],
            "image_mode": image_info["mode"],
            "requires_conversion": False,
            "message": "Image validation passed.",
        }

    def _validate_filename(self, filename: str) -> None:
        if not filename or not filename.strip():
            raise ValueError("Filename is missing")

    def _validate_format(self, extension: str) -> None:
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                "Supported formats are DICOM, JPEG, PNG, and TIFF."
            )

    def _validate_size(self, extension: str, content: bytes) -> None:
        if not content:
            raise ValueError("Uploaded file is empty")

        if len(content) < 32:
            raise ValueError("Uploaded file is too small to contain valid image data")

        if extension == ".dcm":
            if len(content) > self.DICOM_SIZE_LIMIT_BYTES:
                raise ValueError("DICOM upload exceeds 500 MB limit")
            return

        if len(content) > self.NON_DICOM_SIZE_LIMIT_BYTES:
            raise ValueError("Non-DICOM upload exceeds 50 MB limit")

    def _validate_header_integrity(self, extension: str, content: bytes) -> None:
        if extension in {".jpg", ".jpeg"}:
            is_jpeg = len(content) >= 2 and content[:2] == JPEG_MAGIC
            is_png_disguised = len(content) >= 8 and content[:8] == PNG_MAGIC

            if not is_jpeg and not is_png_disguised:
                raise ValueError("Invalid JPEG header")

        elif extension == ".png":
            if len(content) < 8 or content[:8] != PNG_MAGIC:
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

            if content[128:132] != b"DICM":
                raise ValueError("Invalid DICOM header")

    def _validate_content_type(self, extension: str, content_type: str | None) -> None:
        if content_type is None:
            return

        if not isinstance(content_type, str):
            raise ValueError("Unreadable content type metadata")

        normalized = content_type.lower()

        if "application/octet-stream" in normalized:
            return

        if extension in {".jpg", ".jpeg"}:
            if "jpeg" not in normalized and "jpg" not in normalized and "png" not in normalized:
                raise ValueError("JPEG extension does not match content type")

        elif extension == ".png":
            if "png" not in normalized:
                raise ValueError("PNG extension does not match content type")

        elif extension in {".tif", ".tiff"}:
            if "tiff" not in normalized and "tif" not in normalized:
                raise ValueError("TIFF extension does not match content type")

        elif extension == ".dcm":
            dicom_like = (
                "dicom" in normalized
                or "application/dicom" in normalized
                or "application/octet-stream" in normalized
            )

            if not dicom_like:
                raise ValueError("DICOM extension does not match content type")

    def _validate_image_readability(self, content: bytes) -> dict:
        try:
            image = Image.open(BytesIO(content))
            image = ImageOps.exif_transpose(image)
            image.verify()

            image = Image.open(BytesIO(content))
            image = ImageOps.exif_transpose(image)

            width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError("Image has invalid dimensions")

            return {
                "width": width,
                "height": height,
                "mode": image.mode,
            }

        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file could not be read as an image") from exc
        except Exception as exc:
            raise ValueError(f"Uploaded image failed readability check: {str(exc)}") from exc