from fastapi import APIRouter, HTTPException

from app.schemas import TrainJobStatus, TrainRequest
from app.services import train_service
from app.services.train_service import TrainError

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("", response_model=TrainJobStatus)
def start_training(req: TrainRequest):
    try:
        return train_service.start_training(req)
    except TrainError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs", response_model=list[TrainJobStatus])
def list_jobs():
    return train_service.list_jobs()


@router.get("/jobs/{run_name}", response_model=TrainJobStatus)
def get_job(run_name: str):
    try:
        return train_service.get_job(run_name)
    except TrainError as e:
        raise HTTPException(status_code=404, detail=str(e))
