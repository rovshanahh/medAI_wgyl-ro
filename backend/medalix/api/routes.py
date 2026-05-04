from fastapi import APIRouter, File, HTTPException, UploadFile

from medalix.api.orchestrator import Orchestrator
from medalix.inference.ensemble_model import EnsembleModel

router = APIRouter()
orchestrator = Orchestrator()


SUPPORTED_UPLOADS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"]

ACTIVE_ROUTES = [
    {
        "route": "skin_dermoscopy",
        "region": "skin",
        "modality": "dermoscopy",
        "model": "skin_dermoscopy_resnet18",
        "description": "Reviews skin dermoscopy images and returns the most likely lesion class.",
        "status": "ACTIVE",
    },
    {
        "route": "bone_xray",
        "region": "bone",
        "modality": "xray",
        "model": "bone_xray_standard",
        "description": "Reviews bone X-ray images and separates normal from abnormal cases.",
        "status": "ACTIVE",
    },
    {
        "route": "chest_xray",
        "region": "chest",
        "modality": "xray",
        "model": "chest_xray_mvp",
        "description": "Reviews chest X-ray images and highlights possible visible findings.",
        "status": "ACTIVE",
    },
    {
        "route": "retina_fundus",
        "region": "retina",
        "modality": "fundus",
        "model": "retina_fundus_resnet18",
        "description": "Reviews eye fundus images and estimates diabetic retinopathy severity.",
        "status": "ACTIVE",
    },
]

SAFETY_ROUTES = [
    {
        "route": "unknown",
        "region": None,
        "modality": None,
        "model": None,
        "description": "Stops unsupported or uncertain images before inference.",
        "status": "STOP_BEFORE_INFERENCE",
    }
]

INACTIVE_PLACEHOLDERS = [
    {
        "route": "abdomen_ct",
        "region": "abdomen",
        "modality": "ct",
        "status": "INACTIVE",
    },
    {
        "route": "breast_mammography",
        "region": "breast",
        "modality": "mammography",
        "status": "INACTIVE",
    },
    {
        "route": "skin_dermoscopy",
        "region": "skin",
        "modality": "dermoscopy",
        "status": "INACTIVE",
    },
]


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MedAIx backend",
    }


@router.get("/")
def root():
    return {
        "message": "MedAIx backend is running",
        "supported_uploads": SUPPORTED_UPLOADS,
        "routes": [route["route"] for route in ACTIVE_ROUTES] + ["unknown"],
    }


@router.get("/config")
def config():
    return {
        "service": "MedAIx backend",
        "supported_uploads": SUPPORTED_UPLOADS,
        "active_routes": ACTIVE_ROUTES,
        "safety_routes": SAFETY_ROUTES,
        "inactive_placeholders": INACTIVE_PLACEHOLDERS,
        "disclaimer": (
            "Research-use only. Outputs are non-diagnostic and must not be used "
            "for clinical decision-making."
        ),
    }


@router.get("/routes")
def routes():
    return {
        "active_routes": ACTIVE_ROUTES,
        "safety_routes": SAFETY_ROUTES,
        "inactive_placeholders": INACTIVE_PLACEHOLDERS,
    }


@router.get("/model-cache")
def model_cache():
    return EnsembleModel.cache_info()


@router.post("/model-cache/clear")
def clear_model_cache():
    EnsembleModel.clear_cache()
    return {
        "status": "cleared",
        "cache": EnsembleModel.cache_info(),
    }


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()

    return orchestrator.execute(
        filename=file.filename or "upload",
        content_type=file.content_type,
        content=content,
    )


@router.get("/result/{analysis_id}")
def get_result(analysis_id: str):
    result = orchestrator.get_result(analysis_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found")

    return result