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
        "routes": ["brain_mri", "bone_xray", "chest_xray", "unknown"],
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