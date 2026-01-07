# planparser/api.py
import os
import io
from glob import glob

from ultralytics import YOLO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR")
if not MODEL_DIR:
    raise RuntimeError("MODEL_DIR is not set")
MODEL_DIR = os.path.abspath(os.path.expanduser(MODEL_DIR))

app = FastAPI(title="Model API")
_models: dict[str, YOLO] = {}


def _pick_weights(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"model not found: {path}")

    for name in ["model.pt", "best.pt"]:
        cand = os.path.join(path, name)
        if os.path.isfile(cand):
            return cand

    pts = sorted(glob(os.path.join(path, "*.pt")))
    if pts:
        return pts[0]

    raise HTTPException(status_code=400, detail=f"no .pt weights in: {path}")


def _resolve_weights_path(model_name: str) -> str:
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    p = os.path.abspath(os.path.join(MODEL_DIR, model_name))
    if os.path.commonpath([MODEL_DIR, p]) != MODEL_DIR:
        raise HTTPException(status_code=400, detail="model_name must be inside MODEL_DIR")

    return _pick_weights(p)


def load_model(weights_path: str) -> YOLO:
    weights_path = os.path.abspath(weights_path)
    m = _models.get(weights_path)
    if m is None:
        m = YOLO(weights_path)
        _models[weights_path] = m
    return m


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]


class PredictResponse(BaseModel):
    detections: list[Detection]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...),
):
    weights_path = _resolve_weights_path(model_name)

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    r0 = load_model(weights_path)(img)[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        raise HTTPException(status_code=500, detail="model output has no boxes (not a detection model?)")

    names = getattr(r0, "names", {}) or {}
    dets: list[Detection] = []
    for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
        cls_id = int(cls_t.item())
        conf = float(conf_t.item())
        xyxy = [float(x) for x in xyxy_t.tolist()]
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        dets.append(Detection(class_id=cls_id, class_name=cls_name, confidence=conf, xyxy=xyxy))

    return PredictResponse(detections=dets)
# uv run uvicorn planparser.api:app --host 0.0.0.0 --port 8000