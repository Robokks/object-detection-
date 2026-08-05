import json
import threading
import traceback

from app.config import RUNS_DIR, WEIGHTS_DIR, DEFAULT_PRETRAINED_WEIGHTS
from app.schemas import TrainJobStatus, TrainRequest
from app.services import dataset_service
from app.services.dataset_service import DatasetError

_jobs: dict[str, TrainJobStatus] = {}
_lock = threading.Lock()


class TrainError(ValueError):
    pass


def list_jobs() -> list[TrainJobStatus]:
    with _lock:
        return list(_jobs.values())


def get_job(run_name: str) -> TrainJobStatus:
    with _lock:
        job = _jobs.get(run_name)
    if not job:
        raise TrainError(f"Training run '{run_name}' not found")
    return job


def _set_job(run_name: str, **updates) -> None:
    with _lock:
        job = _jobs[run_name]
        for key, value in updates.items():
            setattr(job, key, value)


def start_training(req: TrainRequest) -> TrainJobStatus:
    with _lock:
        if req.run_name in _jobs and _jobs[req.run_name].state == "running":
            raise TrainError(f"Training run '{req.run_name}' is already running")
        job = TrainJobStatus(
            run_name=req.run_name,
            state="queued",
            mode=req.mode,
            total_epochs=req.epochs,
        )
        _jobs[req.run_name] = job

    thread = threading.Thread(target=_run_training, args=(req,), daemon=True)
    thread.start()
    return job


def _run_training(req: TrainRequest) -> None:
    from ultralytics import YOLO  # imported lazily: heavy dependency, slow to import

    try:
        _set_job(req.run_name, state="running", message="Preparing dataset")
        data_yaml = dataset_service.export_yolo_dataset(req.dataset_name)
        meta = dataset_service.get_dataset(req.dataset_name)
        classes = meta["classes"]

        if req.mode == "finetune":
            checkpoint = f"{req.base_model}.pt" if req.base_model else DEFAULT_PRETRAINED_WEIGHTS
            model = YOLO(checkpoint)
        else:
            config = f"{req.base_model}.yaml" if req.base_model else "yolov8n.yaml"
            model = YOLO(config)

        def on_epoch_end(trainer):
            epoch = trainer.epoch + 1
            total = trainer.epochs
            metrics = {
                k: float(v)
                for k, v in (trainer.metrics or {}).items()
                if isinstance(v, (int, float))
            }
            _set_job(
                req.run_name,
                current_epoch=epoch,
                total_epochs=total,
                progress=round(epoch / total, 4) if total else 0.0,
                message=f"Epoch {epoch}/{total}",
                metrics=metrics,
            )

        model.add_callback("on_train_epoch_end", on_epoch_end)

        _set_job(req.run_name, message="Training started")
        model.train(
            data=str(data_yaml),
            epochs=req.epochs,
            imgsz=req.image_size,
            batch=req.batch_size,
            project=str(RUNS_DIR),
            name=req.run_name,
            exist_ok=True,
            pretrained=(req.mode == "finetune"),
        )

        best_weights = RUNS_DIR / req.run_name / "weights" / "best.pt"
        if not best_weights.exists():
            raise TrainError("Training finished but no weights were produced")

        final_weights = WEIGHTS_DIR / f"{req.run_name}.pt"
        final_weights.write_bytes(best_weights.read_bytes())
        (WEIGHTS_DIR / f"{req.run_name}.classes.json").write_text(json.dumps(classes))

        _set_job(
            req.run_name,
            state="completed",
            progress=1.0,
            message="Training complete",
            weights_path=str(final_weights),
        )
    except DatasetError as e:
        _set_job(req.run_name, state="failed", message=str(e))
    except Exception:
        _set_job(req.run_name, state="failed", message=traceback.format_exc(limit=3))
