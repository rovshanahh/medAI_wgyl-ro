from pathlib import Path
from datetime import datetime

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


SOURCE_IMAGE = Path("test_samples/chest_xray.png")
OUTPUT_DICOM = Path("test_samples/test_chest_xray.dcm")


def main() -> None:
    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError(f"Missing source image: {SOURCE_IMAGE}")

    image = Image.open(SOURCE_IMAGE).convert("L").resize((224, 224))
    pixel_array = np.array(image).astype(np.uint16)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(
        str(OUTPUT_DICOM),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    now = datetime.now()

    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"
    ds.Modality = "CR"
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    ds.PixelData = pixel_array.tobytes()

    OUTPUT_DICOM.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(OUTPUT_DICOM)

    print(f"Saved test DICOM to: {OUTPUT_DICOM}")


if __name__ == "__main__":
    main()