---
sdk: docker
app_port: 7860
---
# 🧩 planparser
**Architectural plan elements detection** based on **Ultralytics YOLO** and **Faster R-CNN (TorchScript)** with a convenient **Gradio UI** and **FastAPI** API.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](#)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-black)](#)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-teal)](#)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](#)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](#)

---

## ✨ What it is
planparser takes a floor plan image, runs it through the selected model (YOLO or Faster R-CNN), and shows the result in a convenient UI:

- 🖼️ image with bboxes and class labels
- 📋 schedule table (Element, Qty)
- ⬇️ downloadable CSV file (element_schedule.csv)
- 🧾 raw detections in JSON (class_id, class_name, confidence, xyxy)
- ⏱️ processing time

---

## 🍬 Features
- 🖼️ image upload + examples
- 🧠 model selection from list
- ⚡ auto-run or manual Submit
- 🖍️ bbox rendering + normalized class names
- 📋 element schedule in a table (Element, Qty)
- ⬇️ schedule export to CSV
- 📦 raw detections in JSON (in Accordion)
- 🧰 FastAPI inference API
- 🐳 containerization and run via Docker
- 🤗 access to the already deployed app on HF Spaces

---

## 🤗 Web app
Ready-to-use web version without local setup:

[**Hugging Face Spaces**](https://huggingface.co/spaces/Ann-Grabetski/planparser)

---

## 🧠 Architecture
````mermaid
flowchart LR
  A[Gradio UI] -->|POST image + weights_path + model_type + conf| B[FastAPI /predict]
  B --> C{Model type}
  C -->|yolo| D[Ultralytics YOLO]
  C -->|fasterrcnn| E[TorchScript Faster R-CNN]
  D --> B
  E --> B
  B -->|detections JSON| A
  A --> F[Render bbox + labels]
  A --> G[Element schedule table]
  A --> H[CSV export]
  A --> I[Raw detections accordion]
````

---

## 🗂️ Project structure

````text
planparser/
  app.py          # Gradio UI client
  api.py          # FastAPI inference server
src/
  examples/       # image examples (optional)
  models/         # *.pt weights (optional)
.env              # config
````

---

## 🚀 Quick start

Installation:

```bash
git clone https://github.com/anngrrr/planparser.git
cd planparser
uv sync
````

---

## ⚙️ Config (.env)

Minimum:

````env
API_URL="http://127.0.0.1:8000"
MODEL_DIR="src/models"
MODEL_1="yolo11l_custom.pt"
MODEL_2="fasterrcnn_resnet50.pt"
EXAMPLES_DIR="src/examples"
````

Table:

| Variable             | Purpose               |
| -------------------- | --------------------- |
| `API_URL`            | FastAPI address for UI |
| `MODEL_DIR`          | weights folder        |
| `MODEL_1`, `MODEL_2` | *.pt file names       |
| `EXAMPLES_DIR`       | examples folder for UI |

---

## 🏃 Run locally

### 1) Start API

````bash
uv run uvicorn planparser.api:app --host 0.0.0.0 --port 8000
````

### 2) Start UI

````bash
uv run python planparser/app.py
````

Open:

* API: `http://127.0.0.1:8000`
* UI: `http://127.0.0.1:7860`

---
## 🔌 API

### `GET /health`
Response:
````json
{"ok": true}
````

### `POST /predict`

Form-data:

* `file`: image
* `weights_path`: path to `.pt` weights file (must exist on the API side)
* `model_type` (optional): `yolo` or `fasterrcnn` (default `yolo`)
* `conf` (optional): confidence threshold (default `0.25`)

Response example:

````json
{
  "detections": [
    {
      "class_id": 1,
      "class_name": "door",
      "confidence": 0.87,
      "xyxy": [12.3, 45.6, 78.9, 120.1]
    }
  ]
}
````

---

## 🧩 Models

### Local weights

Put `.pt` files into `MODEL_DIR` and specify them in `.env`:

````env
MODEL_DIR="src/models"
MODEL_1="yolo11l_custom.pt"
MODEL_2="fasterrcnn_resnet50.pt"
````

### How weights are selected
Weights are selected in the UI and sent to the API as `weights_path`.

UI:
- reads `MODEL_DIR`
- builds the list of available models from `.env` (`MODEL_1`, `MODEL_2`)
- shows them in the Dropdown

API:
- accepts `weights_path`
- checks that it is an existing `.pt` file
- caches loaded models by absolute path (to avoid reloading weights)

---

## 🐳 Docker

### Build

````bash
docker build -t planparser .
````

### Run

````bash
docker run --rm \
  -p 7860:7860 -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/src/models:/app/src/models" \
  -v "$(pwd)/src/examples:/app/src/examples" \
  planparser
````

---

## 🧪 Training (if needed)

Minimal example (Ultralytics):

````bash
yolo detect train model=yolo11n.pt data=src/data/data.yaml imgsz=640 epochs=50
````

For Faster R-CNN see notebook: `notebooks/05_train-fasterrcnn-resnet50.ipynb`.

---

## 📎 Datasets and licenses

### Dataset
Uses dataset [**Floorplan details Fork**](https://universe.roboflow.com/research-g8szb/floorplan-details-fork/dataset/1), license **CC BY 4.0**

### Ultralytics YOLO

Ultralytics YOLO is distributed under **AGPL-3.0**

---

## ❤️ Credits

* Ultralytics YOLO
* Gradio
* FastAPI
* Hugging Face

---

## 📜 License

See `LICENSE` file.
