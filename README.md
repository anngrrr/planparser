---
sdk: docker
app_port: 7860
---
# 🧩 planparser
**Architectural plan elements detection** на базе **Ultralytics YOLO** и **Faster R-CNN (TorchScript)** с удобным **Gradio UI** и **FastAPI** API.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](#)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-black)](#)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-teal)](#)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](#)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](#)

---

## ✨ Что это
planparser берет изображение плана, прогоняет через выбранную модель (YOLO или Faster R-CNN) и показывает результат в удобном UI:

- 🖼️ изображение с bbox и подписями классов
- 📋 таблицу-спецификацию (Element, Qty)
- ⬇️ CSV файл для скачивания (element_schedule.csv)
- 🧾 raw detections в JSON (class_id, class_name, confidence, xyxy)
- ⏱️ время обработки

---

## 🍬 Фичи
- 🖼️ Upload изображения + Examples
- 🧠 выбор модели из списка
- ⚡ Auto-run или ручной Submit
- 🖍️ отрисовка bbox + нормализованные названия классов
- 📋 element schedule в таблице (Element, Qty)
- ⬇️ экспорт schedule в CSV
- 📦 raw detections в JSON (в Accordion)
- 🧰 FastAPI inference API
- 🐳 контейнеризация и запуск через Docker
- 🤗 доступ к уже развернутому приложению на HF Spaces

---

## 🤗 Web app
Готовая веб-версия без локальной установки:

[**Hugging Face Spaces**](https://huggingface.co/spaces/Ann-Grabetski/planparser)

---

## 🧠 Архитектура
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

## 🗂️ Структура проекта

````text
planparser/
  app.py          # Gradio UI клиент
  api.py          # FastAPI инференс сервер
src/
  examples/       # примеры картинок (опционально)
  models/         # веса *.pt (опционально)
.env              # конфиг
````

---

## 🚀 Быстрый старт

Установка:

```bash
git clone https://github.com/anngrrr/planparser.git
cd planparser
uv sync
````

---

## ⚙️ Конфиг (.env)

Минимум:

````env
API_URL="http://127.0.0.1:8000"
MODEL_DIR="src/models"
MODEL_1="yolo11l_custom.pt"
MODEL_2="fasterrcnn_resnet50.pt"
EXAMPLES_DIR="src/examples"
````

Таблица:

| Переменная           | Зачем                 |
| -------------------- | --------------------- |
| `API_URL`            | адрес FastAPI для UI  |
| `MODEL_DIR`          | папка с весами        |
| `MODEL_1`, `MODEL_2` | имена файлов *.pt     |
| `EXAMPLES_DIR`       | папка примеров для UI |

---

## 🏃 Запуск локально

### 1) Поднять API

````bash
uv run uvicorn planparser.api:app --host 0.0.0.0 --port 8000
````

### 2) Поднять UI

````bash
uv run python planparser/app.py
````

Открыть:

* API: `http://127.0.0.1:8000`
* UI: `http://127.0.0.1:7860`

---
## 🔌 API

### `GET /health`
Ответ:
````json
{"ok": true}
````

### `POST /predict`

Form-data:

* `file`: изображение
* `weights_path`: путь к `.pt` файлу весов (должен существовать на стороне API)
* `model_type` (опционально): `yolo` или `fasterrcnn` (по умолчанию `yolo`)
* `conf` (опционально): порог confidence (по умолчанию `0.25`)

Пример ответа:

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

## 🧩 Модели

### Локальные веса

Положи `.pt` в `MODEL_DIR` и укажи в `.env`:

````env
MODEL_DIR="src/models"
MODEL_1="yolo11l_custom.pt"
MODEL_2="fasterrcnn_resnet50.pt"
````

### Как выбираются веса
Веса выбираются в UI и передаются в API как `weights_path`.

UI:
- берет `MODEL_DIR`
- собирает список доступных моделей из `.env` (`MODEL_1`, `MODEL_2`)
- отображает их в Dropdown

API:
- принимает `weights_path`
- проверяет что это существующий `.pt` файл
- кэширует загруженные модели по абсолютному пути (чтобы повторно не грузить веса)

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

## 🧪 Трейнинг (если надо)

Минимальный пример (Ultralytics):

````bash
yolo detect train model=yolo11n.pt data=src/data/data.yaml imgsz=640 epochs=50
````

Для Faster R-CNN см. notebook: `notebooks/05_train-fasterrcnn-resnet50.ipynb`.

---

## 📎 Датасеты и лицензии

### Dataset
Используется датасет [**Floorplan details Fork**](https://universe.roboflow.com/research-g8szb/floorplan-details-fork/dataset/1), лицензия **CC BY 4.0**

### Ultralytics YOLO

Ultralytics YOLO распространяется по **AGPL-3.0**

---

## ❤️ Credits

* Ultralytics YOLO
* Gradio
* FastAPI
* Hugging Face

---

## 📜 License

Смотри файл `LICENSE`.
