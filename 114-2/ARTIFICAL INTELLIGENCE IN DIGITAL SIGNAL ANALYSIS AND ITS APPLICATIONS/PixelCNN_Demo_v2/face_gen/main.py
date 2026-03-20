import os
import io
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

# =========================
# 基本設定
# =========================
APP_TITLE = "PixelCNN Demo"
APP_SUBTITLE = "Dataset: LFW"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pixelcnn_lfw.onnx"
OUTPUT_DIR = BASE_DIR / "outputs"

IMG_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 256
NUM_SAMPLES = 8

DATA_MEAN = 0.5
DATA_STD = 0.5

OUTPUT_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"找不到 ONNX 模型：{MODEL_PATH}")

# =========================
# 載入 ONNX
# =========================
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# =========================
# 工具函式
# =========================
def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sample_from_logits(logits_1d: np.ndarray) -> int:
    probs = softmax_np(logits_1d.astype(np.float64), axis=-1)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_faces_onnx(num_samples: int = NUM_SAMPLES, progress=gr.Progress()) -> list[np.ndarray]:
    """
    使用 ONNX 版 PixelCNN 逐像素生成 RGB 人臉
    回傳 list[np.ndarray]，每張 shape=(64,64,3), dtype=uint8
    """
    generated = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.int32)

    total_steps = IMG_SIZE * IMG_SIZE
    step = 0

    for h in range(IMG_SIZE):
        for w in range(IMG_SIZE):
            x_0_1 = generated.astype(np.float32) / 255.0
            x_norm = (x_0_1 - DATA_MEAN) / DATA_STD

            logits = session.run([output_name], {input_name: x_norm})[0]
            logits = logits.reshape(num_samples, IMG_SIZE, IMG_SIZE, CHANNELS, NUM_CLASSES)

            for n in range(num_samples):
                for c in range(CHANNELS):
                    pixel_logits = logits[n, h, w, c, :]
                    sampled_value = sample_from_logits(pixel_logits)
                    generated[n, h, w, c] = sampled_value

            step += 1
            if step % 32 == 0 or step == total_steps:
                progress(step / total_steps, desc=f"生成中... {step}/{total_steps} pixels")

    return [generated[i].astype(np.uint8) for i in range(num_samples)]


def make_zip(images: list[np.ndarray]) -> str:
    """
    將生成圖片存成 PNG 並打包成 ZIP，回傳 ZIP 路徑
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = OUTPUT_DIR / f"pixelcnn_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    zip_path = OUTPUT_DIR / f"pixelcnn_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, img in enumerate(images, start=1):
            img_path = save_dir / f"generated_{idx:02d}.png"
            Image.fromarray(img).save(img_path)
            zf.write(img_path, arcname=img_path.name)

    return str(zip_path)


# =========================
# Gradio 事件函式
# =========================
def on_generate():
    images = generate_faces_onnx(NUM_SAMPLES)
    gallery_images = [Image.fromarray(img) for img in images]
    status = "✅ 已生成 8 張圖片"
    zip_path = make_zip(images)
    return gallery_images, images, status, gr.update(value=zip_path, visible=True)


def on_save(images_state):
    if not images_state:
        return "目前沒有可儲存的圖片，請先按「生成圖片」。", gr.update(visible=False)

    zip_path = make_zip(images_state)
    return "✅ 圖片已儲存並打包完成", gr.update(value=zip_path, visible=True)


# =========================
# 建立 UI
# =========================
with gr.Blocks(title=APP_TITLE) as demo:
    images_state = gr.State([])

    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(f"## {APP_SUBTITLE}")
    gr.Markdown("⚠️ PixelCNN 為逐像素生成模型，生成時間較長，請耐心等待。")

    with gr.Row():
        generate_btn = gr.Button("生成圖片", variant="primary")
        save_btn = gr.Button("儲存圖片")

    status_box = gr.Textbox(label="狀態", interactive=False, value="就緒")

    gallery = gr.Gallery(
        label="生成結果",
        columns=4,
        rows=2,
        height="auto",
        preview=True
    )

    download_btn = gr.DownloadButton(
        label="下載 ZIP 檔",
        value=None,
        visible=False
    )

    generate_btn.click(
        fn=on_generate,
        inputs=[],
        outputs=[gallery, images_state, status_box, download_btn],
        api_name=False
    )

    save_btn.click(
        fn=on_save,
        inputs=[images_state],
        outputs=[status_box, download_btn],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )