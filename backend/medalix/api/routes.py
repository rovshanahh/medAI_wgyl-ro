from fastapi import APIRouter, File, HTTPException, UploadFile

from medalix.api.orchestrator import Orchestrator
from medalix.inference.ensemble_model import EnsembleModel

router = APIRouter()
orchestrator = Orchestrator()


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
        "supported_uploads": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"],
        "routes": ["brain_mri", "bone_xray", "chest_xray", "retina_fundus", "unknown"],
    }


@router.get("/routes")
def routes():
    return {
        "active_routes": [
            {
                "route": "brain_mri",
                "region": "brain",
                "modality": "mri",
                "model": "brain_mri_resnet18",
                "status": "ACTIVE",
            },
            {
                "route": "bone_xray",
                "region": "bone",
                "modality": "xray",
                "model": "bone_xray_standard",
                "status": "ACTIVE",
            },
            {
                "route": "chest_xray",
                "region": "chest",
                "modality": "xray",
                "model": "chest_xray_mvp",
                "status": "ACTIVE",
            },
            {
                "route": "retina_fundus",
                "region": "retina",
                "modality": "fundus",
                "model": "retina_fundus_resnet18",
                "status": "ACTIVE",
            },
        ],
        "safety_routes": [
            {
                "route": "unknown",
                "region": None,
                "modality": None,
                "model": None,
                "status": "STOP_BEFORE_INFERENCE",
            }
        ],
        "inactive_placeholders": [
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
        ],
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