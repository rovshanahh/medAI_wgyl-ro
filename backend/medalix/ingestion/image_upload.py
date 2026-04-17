from dataclasses import dataclass


@dataclass(frozen=True)
class ImageUpload:
    filename: str
    content_type: str | None
    content: bytes

    @property
    def size_bytes(self) -> int:
        return len(self.content)

    @property
    def extension(self) -> str:
        dot_index = self.filename.rfind(".")
        if dot_index == -1:
            return ""
        return self.filename[dot_index:].lower()