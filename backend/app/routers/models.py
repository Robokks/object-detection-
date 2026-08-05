from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import ModelInfo, ModelTask
from app.services import model_service
from app.services.model_service import ModelError

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models():
    return model_service.list_models()


@router.post("/import", response_model=ModelInfo)
async def import_model(
    file: UploadFile = File(...),
    label: str = Form(""),
    task: ModelTask = Form("detect"),
):
    content = await file.read()
    try:
        # Validating the checkpoint means loading it, which can take a
        # while for larger models — keep it off the event loop.
        return await run_in_threadpool(
            model_service.import_model, file.filename or "model.pt", content, label, task
        )
    except ModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
