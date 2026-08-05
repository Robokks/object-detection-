from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import dataset, detect, models, train

app = FastAPI(title="Vision Object Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset.router)
app.include_router(train.router)
app.include_router(models.router)
app.include_router(detect.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
