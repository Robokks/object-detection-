from fastapi import APIRouter

from app.schemas import ModelInfo
from app.services import model_service

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models():
    return model_service.list_models()
