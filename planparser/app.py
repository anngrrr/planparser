import os
import gradio as gr

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_FILE = os.getenv("MODEL_FILE", "model.pt")

_model = None


def get_model_path() -> str:
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}. Put the file there or set MODEL_DIR and MODEL_FILE."
        )
    return path


def load_model():
    global _model
    if _model is not None:
        return _model
    import torch

    _model = torch.jit.load(get_model_path(), map_location="cpu").eval()
    return _model


def predict(text: str) -> str:
    _ = load_model()
    return f"Loaded local model: {os.path.join(MODEL_DIR, MODEL_FILE)}. Input: {text}"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Textbox(label="Output"),
    title="Minimal demo",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
