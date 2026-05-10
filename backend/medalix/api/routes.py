from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from medalix.api.orchestrator import Orchestrator
from medalix.config.route_metadata import (
    ACTIVE_ROUTE_METADATA,
    INACTIVE_PLACEHOLDERS,
    SAFETY_ROUTES,
)
from medalix.inference.ensemble_model import EnsembleModel


router = APIRouter()
orchestrator = Orchestrator()

SUPPORTED_UPLOADS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"]
MAX_BATCH_SIZE = 25


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
        "routes": [route["route"] for route in ACTIVE_ROUTE_METADATA] + ["unknown"],
        "max_batch_size": MAX_BATCH_SIZE,
    }


@router.get("/config")
def config():
    return {
        "service": "MedAIx backend",
        "supported_uploads": SUPPORTED_UPLOADS,
        "active_routes": ACTIVE_ROUTE_METADATA,
        "safety_routes": SAFETY_ROUTES,
        "inactive_placeholders": INACTIVE_PLACEHOLDERS,
        "max_batch_size": MAX_BATCH_SIZE,
        "disclaimer": (
            "Research-use only. Outputs are non-diagnostic and must not be used "
            "for clinical decision-making."
        ),
    }


@router.get("/routes")
def routes():
    return {
        "active_routes": ACTIVE_ROUTE_METADATA,
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


@router.post("/analyze/override")
async def analyze_image_with_override(
    file: UploadFile = File(...),
    route_override: str = Form(...),
):
    content = await file.read()

    return orchestrator.execute(
        filename=file.filename or "upload",
        content_type=file.content_type,
        content=content,
        route_override=route_override,
    )


@router.post("/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch limit exceeded. Maximum allowed files: {MAX_BATCH_SIZE}",
        )

    results = []

    for index, file in enumerate(files, start=1):
        try:
            content = await file.read()

            result = orchestrator.execute(
                filename=file.filename or f"upload_{index}",
                content_type=file.content_type,
                content=content,
            )

            results.append(
                {
                    "index": index,
                    "filename": file.filename or f"upload_{index}",
                    "status": "completed",
                    "result": result,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "index": index,
                    "filename": file.filename or f"upload_{index}",
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "batch_size": len(files),
        "max_batch_size": MAX_BATCH_SIZE,
        "completed": sum(1 for item in results if item["status"] == "completed"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "results": results,
    }


@router.get("/result/{analysis_id}")
def get_result(analysis_id: str):
    result = orchestrator.get_result(analysis_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found")

    return result