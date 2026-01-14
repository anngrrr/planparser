import os
import io

from ultralytics import YOLO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Planparser API")
_models: dict[str, YOLO] = {}


def load_model(weights_path: str) -> YOLO:
    weights_path = os.path.abspath(os.path.expanduser(weights_path))
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
    weights_path: str = Form(...),
):
    weights_path = os.path.abspath(os.path.expanduser(weights_path))
    if not (os.path.isfile(weights_path) and weights_path.lower().endswith(".pt")):
        raise HTTPException(status_code=400, detail=f"weights_path must be an existing .pt file: {weights_path}")

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
