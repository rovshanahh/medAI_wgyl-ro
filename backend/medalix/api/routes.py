from fastapi import APIRouter, File, HTTPException, UploadFile

from medalix.api.orchestrator import Orchestrator
from medalix.config.route_metadata import (
    ACTIVE_ROUTE_METADATA,
    INACTIVE_PLACEHOLDERS,
    SAFETY_ROUTES,
)
from medalix.inference.ensemble_model import EnsembleModel
from medalix.report.analysis_report_builder import AnalysisReportBuilder


router = APIRouter()
orchestrator = Orchestrator()


SUPPORTED_UPLOADS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"]


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
    }


@router.get("/config")
def config():
    return {
        "service": "MedAIx backend",
        "supported_uploads": SUPPORTED_UPLOADS,
        "active_routes": ACTIVE_ROUTE_METADATA,
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


@router.get("/result/{analysis_id}")
def get_result(analysis_id: str):
    result = orchestrator.get_result(analysis_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found")

    return result


@router.get("/result/{analysis_id}/report")
def get_result_report(analysis_id: str):
    result = orchestrator.get_result(analysis_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found")

    return AnalysisReportBuilder.build(result)