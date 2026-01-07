# planparser/app.py
import os
import io
import time

import gradio as gr
import requests
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
if not API_URL:
    raise RuntimeError("API_URL is not set")

EXAMPLE_1 = os.getenv("EXAMPLE_1")
EXAMPLE_2 = os.getenv("EXAMPLE_2")

MODEL_MAP = {
    "yolo11n": os.getenv("MODEL_1"),
    "yolo11s": os.getenv("MODEL_2"),
    "yolo11l": os.getenv("MODEL_3"),
}

MODEL_CHOICES = list(MODEL_MAP.keys())
DEFAULT_MODEL = MODEL_CHOICES[0]


def _draw_detections(img: Image.Image, dets: list[dict]) -> Image.Image:
    out = img.copy().convert("RGB")
    d = ImageDraw.Draw(out)

    for det in dets:
        x1, y1, x2, y2 = det["xyxy"]
        d.rectangle([x1, y1, x2, y2], width=2)
        txt = f'{det["class_name"]} {det["confidence"]:.2f}'
        d.text((x1, max(0, y1 - 12)), txt)

    return out


def predict(model_label: str, img: Image.Image):
    if img is None or not model_label:
        return None, [], ""

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)

    t0 = time.perf_counter()
    r = requests.post(
        f"{API_URL}/predict",
        files={"file": ("image.jpg", buf, "image/jpeg")},
        data={"model_name": MODEL_MAP[model_label]},
        timeout=60,
    )
    dt = time.perf_counter() - t0

    if r.status_code != 200:
        return None, {"error": r.text}, f"{dt:.3f} s"

    data = r.json()
    dets = data.get("detections", []) or []
    vis = _draw_detections(img, dets)

    return vis, dets, f"{dt:.3f} s"


with gr.Blocks(title="YOLO detection demo") as demo:
    gr.Markdown("# YOLO detection demo")

    model_dd = gr.Dropdown(
        choices=MODEL_CHOICES,
        value=DEFAULT_MODEL,
        label="Model",
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(
                type="pil",
                label="Image",
                sources=["upload"],
                height=200,
            )

            ex = [p for p in [EXAMPLE_1, EXAMPLE_2] if p]
            if ex:
                gr.Examples(examples=ex, inputs=img_in)

            btn = gr.Button("Submit")

        with gr.Column(scale=2):
            out_img = gr.Image(type="pil", label="Result", height=600)
            out_json = gr.JSON(label="Detections")
            out_time = gr.Textbox(label="Processing time")

    btn.click(predict, inputs=[model_dd, img_in], outputs=[out_img, out_json, out_time])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
# uv run gradio planparser/app.py
