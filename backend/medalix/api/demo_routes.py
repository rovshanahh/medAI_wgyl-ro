from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from medalix.api.orchestrator import Orchestrator


router = APIRouter(prefix="/demo", tags=["demo"])
orchestrator = Orchestrator()


DEMO_CASES = {
    "abdomen_ct": {
        "title": "Valid abdomen CT",
        "description": "Routes to abdomen_ct and classifies kidney CT as Cyst, Normal, Stone, or Tumor.",
        "path": "test_samples/abdomen_ct.jpg",
        "expected": "ANSWER or ESCALATE with abdomen_ct_resnet18 and Grad-CAM++ heatmap.",
    },
    "brain_mri": {
        "title": "Valid brain MRI",
        "description": "Routes to brain_mri and classifies tumor-related MRI class.",
        "path": "test_samples/brain_mri.jpg",
        "expected": "ANSWER or ESCALATE with brain_mri_resnet18 and Grad-CAM++ heatmap.",
    },
    "bone_xray": {
        "title": "Valid bone X-ray",
        "description": "Routes to bone_xray and classifies the musculoskeletal X-ray as Normal or Abnormal.",
        "path": "test_samples/bone_xray.png",
        "expected": "ANSWER or ESCALATE with bone_xray_standard and Grad-CAM++ heatmap.",
        "route_override": "bone_xray",
    },
    "chest_xray": {
        "title": "Valid chest X-ray",
        "description": "Routes to chest_xray and returns CheXpert-style multilabel findings.",
        "path": "test_samples/chest_xray.jpg",
        "expected": "ANSWER or ESCALATE with chest_xray_mvp and Grad-CAM++ heatmap.",
    },
    "breast_mammography": {
        "title": "Breast mammography uncertainty",
        "description": "Routes to breast_mammography and may escalate when confidence is not high enough.",
        "path": "test_samples/breast_mammography.jpg",
        "expected": "Often ESCALATE due to uncertainty, while still showing inference and heatmap.",
    },
    "dicom_chest_xray": {
        "title": "DICOM conversion",
        "description": "Loads a DICOM chest X-ray sample, converts it, routes it, and analyzes it.",
        "path": "test_samples/test_chest_xray.dcm",
        "expected": "DICOM conversion + chest_xray inference + Grad-CAM++ heatmap.",
    },
    "unknown": {
        "title": "Unknown / unsafe input",
        "description": "Uses a random non-medical or unsupported image.",
        "path": "test_samples/random.jpg",
        "expected": "STOP before inference. No model output and no heatmap.",
    },
    "manual_override_random_to_chest": {
        "title": "Manual route confirmation",
        "description": "Starts from an unknown random image but forces chest_xray through controlled override.",
        "path": "test_samples/random.jpg",
        "expected": "manual_override=true and result is auditable.",
        "route_override": "chest_xray",
    },
}


@router.get("/cases")
def list_demo_cases() -> dict:
    return {
        "cases": [
            {
                "id": case_id,
                "title": item["title"],
                "description": item["description"],
                "expected": item["expected"],
                "route_override": item.get("route_override"),
            }
            for case_id, item in DEMO_CASES.items()
        ]
    }


@router.post("/run/{case_id}")
def run_demo_case(case_id: str) -> dict:
    if case_id not in DEMO_CASES:
        raise HTTPException(status_code=404, detail=f"Unknown demo case: {case_id}")

    item = DEMO_CASES[case_id]
    path = Path(item["path"])

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Demo file is missing: {path}",
        )

    result = orchestrator.execute(
        filename=path.name,
        content_type=None,
        content=path.read_bytes(),
        route_override=item.get("route_override"),
    )

    return {
        "case_id": case_id,
        "title": item["title"],
        "description": item["description"],
        "expected": item["expected"],
        "result": result,
    }


@router.get("/image/{case_id}")
def get_demo_image(case_id: str):
    if case_id not in DEMO_CASES:
        raise HTTPException(status_code=404, detail=f"Unknown demo case: {case_id}")

    path = Path(DEMO_CASES[case_id]["path"])

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Demo file is missing: {path}")

    if path.suffix.lower() == ".dcm":
        raise HTTPException(
            status_code=400,
            detail="DICOM preview is not available as a browser image.",
        )

    return FileResponse(path)
