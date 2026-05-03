from io import BytesIO

import numpy as np
import pydicom
from PIL import Image


class DicomConverter:
    def convert_to_png_bytes(self, raw_bytes: bytes) -> bytes:
        try:
            dataset = pydicom.dcmread(BytesIO(raw_bytes), force=False)
        except Exception as exc:
            raise ValueError(f"Failed to read DICOM file: {str(exc)}") from exc

        if not hasattr(dataset, "pixel_array"):
            raise ValueError("DICOM file does not contain an accessible pixel array")

        try:
            pixel_array = dataset.pixel_array.astype(np.float32)
        except Exception as exc:
            raise ValueError(f"Failed to extract DICOM pixel array: {str(exc)}") from exc

        if pixel_array.ndim == 3:
            pixel_array = pixel_array[0]

        if pixel_array.ndim != 2:
            raise ValueError(
                f"Unsupported DICOM pixel array shape: {tuple(pixel_array.shape)}"
            )

        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixel_array = pixel_array * slope + intercept

        photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()

        min_val = float(np.min(pixel_array))
        max_val = float(np.max(pixel_array))

        if max_val <= min_val:
            raise ValueError("DICOM pixel array has no usable intensity range")

        normalized = (pixel_array - min_val) / (max_val - min_val)

        if photometric == "MONOCHROME1":
            normalized = 1.0 - normalized

        image_array = (normalized * 255.0).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_array, mode="L").convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()