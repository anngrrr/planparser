import os
import io
from typing import Any

import torch
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from torchvision.transforms.functional import pil_to_tensor as tv_pil_to_tensor


app = FastAPI(title="Planparser API")

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _abs_pt(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def get_model(model_type: str, weights_path: str):
    wp = _abs_pt(weights_path)
    key = (model_type, wp)

    m = _MODEL_CACHE.get(key)
    if m is not None:
        return m

    if model_type == "yolo":
        m = YOLO(wp)

    elif model_type == "fasterrcnn":
        m = torch.jit.load(wp, map_location="cpu")
        m.eval()

    else:
        raise HTTPException(status_code=400, detail="model_type must be yolo or fasterrcnn")

    _MODEL_CACHE[key] = m
    return m


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return tv_pil_to_tensor(img.convert("RGB")).float().div(255.0)


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
    model_type: str = Form("yolo"),
    conf: float = Form(0.25),
):
    weights_path = _abs_pt(weights_path)
    if not (os.path.isfile(weights_path) and weights_path.lower().endswith(".pt")):
        raise HTTPException(status_code=400, detail=f"weights_path must be an existing .pt file: {weights_path}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="invalid image file")

    if model_type == "yolo":
        model = get_model("yolo", weights_path)
        with torch.inference_mode():
            r0 = model.predict(img, device="cpu", verbose=False)[0]

        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            raise HTTPException(status_code=500, detail="model output has no boxes")

        names = getattr(r0, "names", {}) or {}
        dets: list[Detection] = []
        for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
            cls_id = int(cls_t.item())
            c = float(conf_t.item())
            if c < float(conf):
                continue
            xyxy = [float(x) for x in xyxy_t.tolist()]
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            dets.append(Detection(class_id=cls_id, class_name=cls_name, confidence=c, xyxy=xyxy))

        return PredictResponse(detections=dets)

    if model_type == "fasterrcnn":
        try:
            x = pil_to_tensor(img)
            model = get_model("fasterrcnn", weights_path)

            get_names = getattr(model, "get_class_names", None)
            class_names = list(get_names()) if callable(get_names) else []

            with torch.inference_mode():
                out = model([x])

            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            dets: list[Detection] = []
            for b, s, l in zip(boxes, scores, labels):
                c = float(s)
                if c < float(conf):
                    continue

                cid = int(l)
                if cid == 0:
                    continue

                cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)

                dets.append(
                    Detection(
                        class_id=cid,
                        class_name=cname,
                        confidence=c,
                        xyxy=[float(v) for v in b.tolist()],
                    )
                )

            return PredictResponse(detections=dets)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"fasterrcnn failed: {type(e).__name__}: {e}")

    raise HTTPException(status_code=400, detail="model_type must be yolo or fasterrcnn")
