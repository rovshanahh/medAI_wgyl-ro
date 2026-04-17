from fastapi import APIRouter, File, HTTPException, UploadFile

from medalix.api.orchestrator import Orchestrator

router = APIRouter()
orchestrator = Orchestrator()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/")
def root():
    return {"message": "MedAIx backend is running"}


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return orchestrator.execute(
            filename=file.filename,
            content_type=file.content_type,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(exc)}") from exc


@router.get("/result/{analysis_id}")
def get_result(analysis_id: str):
    result = orchestrator.get_result(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return result