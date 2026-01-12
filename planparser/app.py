# planparser/app.py
import os
import io
import time
import tempfile
import random
from pathlib import Path

import gradio as gr
import requests
import pandas as pd
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
if not API_URL:
    raise RuntimeError("API_URL is not set")

EXAMPLES_DIR = os.getenv("EXAMPLES_DIR")

MODEL_MAP = {
    "yolo11l_custom": os.getenv("MODEL_1"),
    "custom": os.getenv("MODEL_2"),
}
MODEL_MAP = {k: v for k, v in MODEL_MAP.items() if v}

MODEL_CHOICES = list(MODEL_MAP.keys())
if len(MODEL_CHOICES) != 2:
    raise RuntimeError("Exactly two models must be set: MODEL_1 and MODEL_2")

DEFAULT_MODEL = MODEL_CHOICES[0]

_PALETTE = [
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334",
    "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF",
    "FF95C8", "FF37C7",
]

CLASS_NAME_MAP = {
    "bathtub": "Bathtub",
    "bed": "Single bed",
    "bed2": "Double bed",
    "chair": "Chair",
    "door": "Single door",
    "door2": "Double door",
    "shower": "Shower",
    "sink": "Sink",
    "sofa1": "Sofa 1-seater",
    "sofa2": "Sofa 2-seater",
    "sofa3": "Sofa 3-seater",
    "stove": "Stove",
    "table": "Table",
    "toilet": "Toilet",
    "vanity": "Vanity",
}


def _collect_example_images(max_n: int = 30) -> list[str]:
    if not EXAMPLES_DIR:
        return []
    p = Path(EXAMPLES_DIR).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return []

    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    if not files:
        return []

    k = min(max_n, len(files))
    return [str(x) for x in random.sample(files, k=k)]


def _hex2rgb(h: str) -> tuple[int, int, int]:
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _pretty_name(raw: str) -> str:
    return CLASS_NAME_MAP.get(raw, raw)


def _color_for_det(det: dict) -> tuple[int, int, int]:
    cls_id = det.get("class_id", None)
    if cls_id is None:
        name = det.get("class_name", "")
        cls_id = abs(hash(name))
    return _hex2rgb(_PALETTE[int(cls_id) % len(_PALETTE)])


def _draw_detections(img: Image.Image, dets: list[dict]) -> Image.Image:
    out = img.copy().convert("RGB")
    d = ImageDraw.Draw(out)

    for det in dets:
        x1, y1, x2, y2 = det["xyxy"]
        color = _color_for_det(det)

        d.rectangle([x1, y1, x2, y2], outline=color, width=2)

        txt = _pretty_name(det.get("class_name", ""))
        bbox = d.textbbox((x1, y1), txt)
        tx1, ty1, tx2, ty2 = bbox
        pad = 2
        d.rectangle([tx1 - pad, ty1 - pad, tx2 + pad, ty2 + pad], fill=color)
        d.text((x1, y1), txt, fill=(255, 255, 255))

    return out


def _counts_df(dets: list[dict]) -> pd.DataFrame:
    if not dets:
        return pd.DataFrame(columns=["Element", "Qty"])

    df = pd.DataFrame(dets)
    if "class_name" not in df.columns:
        return pd.DataFrame(columns=["Element", "Qty"])

    pretty = df["class_name"].map(_pretty_name)
    out = (
        pretty.value_counts()
        .rename_axis("Element")
        .reset_index(name="Qty")
    )
    return out.sort_values("Element").reset_index(drop=True)

def export_df(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    d = tempfile.mkdtemp(prefix="planparser_")
    path = os.path.join(d, "element_schedule.csv")
    df.to_csv(path, index=False)
    return path


def _request_predict(model_label: str, img: Image.Image) -> tuple[list[dict], float, str | None]:
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
        return [{"error": r.text}], dt, "error"

    data = r.json()
    dets = data.get("detections", []) or []
    return dets, dt, None


def run_predict(model_label: str, img: Image.Image):
    empty_df = _counts_df([])

    if img is None or not model_label:
        return (
            None,
            "",
            empty_df,
            [],
            gr.update(value=None, visible=False),  # out_csv
            gr.update(visible=False),              # raw_acc
        )

    dets, dt, err = _request_predict(model_label, img)
    time_md = f"_processing time: {dt:.3f} s_"

    if err is not None:
        return (
            None,
            time_md,
            empty_df,
            dets,
            gr.update(value=None, visible=False),
            gr.update(visible=False),
        )

    vis = _draw_detections(img, dets)
    df_counts = _counts_df(dets)
    csv_path = export_df(df_counts)

    return (
        vis,
        time_md,
        df_counts,
        dets,
        gr.update(value=csv_path, visible=bool(csv_path)),
        gr.update(visible=bool(dets)),
    )


def maybe_autorun(model_label: str, img: Image.Image, auto_run: bool):
    if not auto_run:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    return run_predict(model_label, img)


with gr.Blocks(title="Architectural plan elements detection") as demo:
    gr.Markdown("# Architectural plan elements detection")

    with gr.Row():
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Model")

            img_in = gr.Image(type="pil", label="Image", sources=["upload"], height=168)

            ex = _collect_example_images(max_n=30)
            if ex:
                gr.Examples(examples=ex, inputs=img_in, label="Examples")

            auto = gr.Checkbox(value=True, label="Auto-run")
            btn = gr.Button("Submit")

        with gr.Column(scale=2):
            out_img = gr.Image(type="pil", label="Result", height=640)
            out_time = gr.Markdown(value="")

            out_df = gr.Dataframe(
                # label="Element schedule",
                headers=["Element", "Qty"],
                datatype=["str", "number"],
                row_count=(0, "dynamic"),
                column_count=(2, "fixed"),
                wrap=True,
                type="pandas",
            )

            out_csv = gr.File(label="Download schedule", visible=False)

            with gr.Accordion("Raw detections", open=False, visible=False) as raw_acc:
                out_json = gr.JSON(label="Detections")

    btn.click(
        run_predict,
        inputs=[model_dd, img_in],
        outputs=[out_img, out_time, out_df, out_json, out_csv, raw_acc],
    )

    img_in.change(
        maybe_autorun,
        inputs=[model_dd, img_in, auto],
        outputs=[out_img, out_time, out_df, out_json, out_csv, raw_acc],
    )
    model_dd.change(
        maybe_autorun,
        inputs=[model_dd, img_in, auto],
        outputs=[out_img, out_time, out_df, out_json, out_csv, raw_acc],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

# uv run gradio planparser/app.py
