from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas import CreateDatasetRequest, DatasetInfo, SaveAnnotationsRequest
from app.services import dataset_service
from app.services.dataset_service import DatasetError

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetInfo])
def list_datasets():
    return dataset_service.list_datasets()


@router.post("", response_model=DatasetInfo)
def create_dataset(req: CreateDatasetRequest):
    try:
        return dataset_service.create_dataset(req)
    except DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{name}")
def get_dataset(name: str):
    try:
        return dataset_service.get_dataset(name)
    except DatasetError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{name}/images")
async def upload_image(name: str, file: UploadFile):
    content = await file.read()
    try:
        return dataset_service.add_image(name, file.filename or "image.jpg", content)
    except DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{name}/images/{image_id}/file")
def get_image_file(name: str, image_id: str):
    try:
        path = dataset_service.image_path(name, image_id)
    except DatasetError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(path)


@router.delete("/{name}/images/{image_id}")
def delete_image(name: str, image_id: str):
    try:
        dataset_service.delete_image(name, image_id)
    except DatasetError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


@router.post("/{name}/annotations")
def save_annotations(name: str, req: SaveAnnotationsRequest):
    try:
        dataset_service.save_annotations(
            name, req.image_id, req.image_width, req.image_height, req.boxes
        )
    except DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}
