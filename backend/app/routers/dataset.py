from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.schemas import CreateDatasetRequest, DatasetInfo, ImportAnnotatedImageResult, SaveAnnotationsRequest, Shape
from app.services import autolabel_service, dataset_service, red_import_service
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


@router.post("/{name}/images/{image_id}/suggest", response_model=list[Shape])
async def suggest_shapes(name: str, image_id: str, class_name: str = Query(...)):
    try:
        path = dataset_service.image_path(name, image_id)
    except DatasetError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing")
    # CV pass over a full-size image can take a moment; keep it off the event loop.
    return await run_in_threadpool(autolabel_service.suggest_cylinder_shapes, path, class_name)


@router.post("/{name}/import-annotated", response_model=list[ImportAnnotatedImageResult])
async def import_annotated_images(
    name: str, files: list[UploadFile] = File(...), class_name: str = Query(...)
):
    results: list[ImportAnnotatedImageResult] = []
    for file in files:
        filename = file.filename or "image.png"
        content = await file.read()
        try:
            clean_bytes, shapes = await run_in_threadpool(
                red_import_service.extract_annotated_image, content, class_name
            )
        except ValueError as e:
            results.append(ImportAnnotatedImageResult(filename=filename, error=str(e)))
            continue

        try:
            added = dataset_service.add_image(name, filename, clean_bytes)
            dataset_service.save_annotations(name, added["image_id"], added["width"], added["height"], shapes)
        except DatasetError as e:
            results.append(ImportAnnotatedImageResult(filename=filename, error=str(e)))
            continue

        results.append(
            ImportAnnotatedImageResult(filename=filename, image_id=added["image_id"], shapes_found=len(shapes))
        )
    return results


@router.post("/{name}/annotations")
def save_annotations(name: str, req: SaveAnnotationsRequest):
    try:
        dataset_service.save_annotations(
            name, req.image_id, req.image_width, req.image_height, req.shapes
        )
    except DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}
