from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import os
import scipy.io.wavfile
import torch
import random
import tempfile
from fastapi import Form
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# 情感分析相關
import tensorflow as tf
from keras.models import load_model
import librosa
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI(title="MusicGen 本地端 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MusicRequest(BaseModel):
    prompt: str
    duration: int

processor = None
music_model = None


def fit_audio_to_duration(audio: np.ndarray, seconds: int, sample_rate: int) -> np.ndarray:
    target_samples = int(seconds * sample_rate)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.shape[0] >= target_samples:
        return audio[:target_samples]
    return np.pad(audio, (0, target_samples - audio.shape[0]))


# ── 情感分析設定 ──────────────────────────────────────────
TAGS = [
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'mellow',
    'dance', '00s', '90s', '80s', '70s', '60s', 'jazz', 'classic rock', 'ambient',
    'experimental', 'instrumental', 'video game music', 'hard rock', 'punk', 'spanish',
    'heavy metal', 'progressive rock', 'acoustic', 'death metal', 'industrial', 'blues',
    'soundtrack', 'british', 'chillout', 'metal', 'vocal', 'classical', 'oldies',
    'hip-hop', 'folk', 'black metal', 'hardcore', 'soul', 'funk', 'new wave',
    'melodic death metal', 'singer-songwriter', 'country', 'house', 'techno',
    'trance', 'reggae', 'electro'
]

TAG_MAP = {
    'house': (0.5, 0.8), 'dance': (0.6, 0.8), 'pop': (0.7, 0.5),
    'heavy metal': (-0.4, 0.9), 'punk': (-0.3, 0.8), 'hardcore': (-0.6, 0.9),
    'ambient': (0.2, -0.7), 'mellow': (0.3, -0.6), 'chillout': (0.4, -0.5),
    'classical': (0.2, -0.3), 'acoustic': (0.3, -0.2), 'jazz': (0.3, -0.1),
    'blues': (-0.3, -0.2), 'folk': (0.2, -0.3), 'soul': (0.4, 0.3),
    'funk': (0.6, 0.6), 'electronic': (0.3, 0.6), 'techno': (0.2, 0.8),
    'trance': (0.4, 0.7), 'rock': (0.1, 0.5), 'indie': (0.2, 0.2),
}
# ─────────────────────────────────────────────────────────


@app.on_event("startup")
async def load_model_on_startup():
    global processor, music_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("📥 正在載入 MusicGen Small 模型...")
    try:
        processor = AutoProcessor.from_pretrained("models/musicgen-small")
        music_model = MusicgenForConditionalGeneration.from_pretrained("models/musicgen-small")
        music_model.to(device)
        print("✅ 模型載入完成，API 已準備就緒！")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return Path("index.html").read_text(encoding="utf-8")


@app.post("/generate-music/")
async def generate_music(request: MusicRequest):
    if request.duration <= 0 or request.duration > 30:
        raise HTTPException(status_code=400, detail="秒數必須大於 0 且小於等於 30 秒")
    if music_model is None or processor is None:
        raise HTTPException(status_code=500, detail="模型尚未載入完成")

    try:
        print(f"🎵 收到純文字生成請求：[{request.prompt}]，長度 {request.duration} 秒...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = processor(
            text=[request.prompt],
            padding=True,
            return_tensors="pt"
        )

        random_seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        audio_values = music_model.generate(
            **inputs.to(device),
            do_sample=True,
            guidance_scale=3,
            max_new_tokens=request.duration * 50
        )

        output_filename = os.path.join(os.getcwd(), f"output_t2m_{random_seed}.wav")
        sampling_rate = music_model.config.audio_encoder.sampling_rate
        audio_out = fit_audio_to_duration(
            audio_values[0, 0].cpu().numpy(),
            request.duration,
            sampling_rate
        )
        scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=audio_out)

        torch.cuda.empty_cache()
        return FileResponse(output_filename, media_type="audio/wav")

    except Exception as e:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-emotion/")
async def analyze_emotion(file: UploadFile = File(...)):
    model_path = "best_model.keras"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="找不到 best_model.keras，請先下載模型")
    
    

    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 1. 讀取並前處理音訊
        #audio, _ = librosa.load(tmp_path, sr=16000)
        #audio_3s = audio[:48000]
        #if len(audio_3s) < 48000:
        #    audio_3s = np.pad(audio_3s, (0, 48000 - len(audio_3s)))
        #
        #mel = librosa.feature.melspectrogram(
        #    y=audio_3s, sr=16000, n_fft=512, hop_length=256, n_mels=96
        #)
        #mel_db = librosa.power_to_db(mel, ref=np.max).T
        #mel_db = np.pad(mel_db, ((0, max(0, 187 - mel_db.shape[0])), (0, 0)))[:187, :]

        # 依照你訓練模型的設定
        SR = 22050
        DURATION = 5
        SAMPLE_SIZE = SR * DURATION
        N_MELS = 128
        N_FFT = 2048
        HOP_LENGTH = 512

        audio, _ = librosa.load(tmp_path, sr=SR, mono=True)
        audio = librosa.util.normalize(audio)

        audio_fixed = audio[:SAMPLE_SIZE]
        if len(audio_fixed) < SAMPLE_SIZE:
            audio_fixed = np.pad(audio_fixed, (0, SAMPLE_SIZE - len(audio_fixed)))

        mel = librosa.feature.melspectrogram(
            y=audio_fixed,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        # 正規化到 [0,1]
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        mel_db = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

        # 確保 shape 是 (128, 216)
        if mel_db.shape[1] < 216:
            mel_db = np.pad(mel_db, ((0, 0), (0, 216 - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :216]


        # 2. 載入 .keras 模型並推理
        model = load_model(model_path)

        # CNN 模型需要 channel 維度
        input_data = mel_db.reshape(1, 128, 216, 1).astype(np.float32)

        preds = model.predict(input_data, verbose=0)[0]

        # 3. 計算加權情感得分
        total_v, total_a, total_w = 0.0, 0.0, 0.0
        top_tags = []
        top_indices = preds.argsort()[-10:][::-1]

        for i in top_indices:
            tag_name = TAGS[i]
            weight = float(max(0, preds[i]))
            top_tags.append({"tag": tag_name, "score": round(weight, 4)})
            if tag_name in TAG_MAP:
                v, a = TAG_MAP[tag_name]
                total_v += v * weight
                total_a += a * weight
                total_w += weight

        final_v = total_v / (total_w + 1e-6)
        final_a = total_a / (total_w + 1e-6)

        # 4. 四象限判定
        if final_v * 100 >= 50 and final_a * 100 >= 50:
            emotion = "Happy"
            emotion_zh = "快樂 / 動感"

        elif final_v * 100 < 50 and final_a * 100 >= 50:
            emotion = "Tensional"
            emotion_zh = "緊張 / 強烈"

        elif final_v * 100 < 50 and final_a * 100 < 50:
            emotion = "Sad"
            emotion_zh = "悲傷 / 憂鬱"

        else:
            emotion = "Peaceful"
            emotion_zh = "平靜 / 輕柔"
        print(final_v)
        print(final_a)
        return {
            "emotion": emotion,
            "emotion_zh": emotion_zh,
            "valence": round(final_v*100, 3),
            "arousal": round(final_a*100, 3),

            "top_tags": top_tags
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"情緒分析失敗: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/generate-music-to-music/")
async def generate_music_to_music(
    prompt: str = Form(...),
    duration: int = Form(...),
    file: UploadFile = File(...)
):
    if duration <= 0 or duration > 30:
        raise HTTPException(status_code=400, detail="秒數必須大於 0 且小於等於 30 秒")
    if music_model is None or processor is None:
        raise HTTPException(status_code=500, detail="模型尚未載入完成")

    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        print(f"🎵 收到 Music-to-Music 請求：[{prompt}]，生成長度 {duration} 秒...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        audio_array, sr = librosa.load(
            tmp_path,
            sr=processor.feature_extractor.sampling_rate
        )

        prompt_length_seconds = 5
        audio_prompt = audio_array[: sr * prompt_length_seconds]

        inputs = processor(
            audio=audio_prompt,
            sampling_rate=sr,
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )

        random_seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        audio_values = music_model.generate(
            **inputs.to(device),
            do_sample=True,
            guidance_scale=3,
            max_new_tokens=duration * 50
        )

        output_filename = os.path.join(os.getcwd(), f"output_m2m_{random_seed}.wav")
        sampling_rate = music_model.config.audio_encoder.sampling_rate
        audio_out = fit_audio_to_duration(
            audio_values[0, 0].cpu().numpy(),
            duration,
            sampling_rate
        )
        scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=audio_out)

        torch.cuda.empty_cache()
        return FileResponse(output_filename, media_type="audio/wav")

    except Exception as e:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)