from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import DetectionResult
from app.services import detect_service
from app.services.model_service import ModelError

router = APIRouter(prefix="/api/detect", tags=["detect"])


@router.post("", response_model=DetectionResult)
async def detect(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    confidence: float = Form(0.25),
):
    content = await file.read()
    try:
        return detect_service.run_detection(model_id, content, confidence)
    except ModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")
