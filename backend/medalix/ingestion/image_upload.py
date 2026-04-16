from dataclasses import dataclass


@dataclass
class ImageUpload:
    filename: str
    content_type: str
    size_bytes: int
    raw_bytes: bytes