import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

# =========================
# Basic settings
# =========================
SR = 48000
DURATION = 4
SAMPLE_SIZE = SR * DURATION
FEATURE_DIM = 1880

RESULTS_DIR = Path('./MER_smile/results')
MODEL_DIR = Path('./MER_smile/model')
TRAIN_ROOT = Path('./MER_smile/song/train')
TEST_ROOT = Path('./MER_smile/song/test')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Happy', 'Tensional', 'Sad', 'Peaceful']
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTS = {'.wav', '.wma'}


# =========================
# Utilities
# =========================
def add_noise(signal: np.ndarray, std: float = 0.05) -> np.ndarray:
    noise = np.random.normal(0, std, signal.shape)
    return signal + noise


def load_audio(audio_path: Path, sr: int = SR) -> np.ndarray:
    """Load wav/wma audio as mono float waveform.
    First try librosa; if it fails, use pydub as fallback.
    """
    try:
        waveform, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        return waveform.astype(np.float32)
    except Exception as e1:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(audio_path))
            audio = audio.set_channels(1).set_frame_rate(sr)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            max_val = max(np.iinfo(audio.array_type).max, 1)
            samples = samples / float(max_val)
            return samples
        except Exception as e2:
            raise RuntimeError(f'Failed to load {audio_path}. librosa: {e1}; pydub: {e2}')


def extract_feature_vector(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=1)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_bands=1)
    rms = librosa.feature.rms(y=signal)

    feature_combine = np.hstack((
        rms.flatten(),
        mfcc.flatten(),
        rolloff.flatten(),
        contrast.flatten()
    )).astype(np.float32)

    if feature_combine.shape[0] != FEATURE_DIM:
        raise ValueError(
            f'Feature dimension mismatch. Expected {FEATURE_DIM}, got {feature_combine.shape[0]}. '
            'Please check librosa parameters or sampling settings.'
        )

    return feature_combine


def iter_audio_files(root: Path):
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            print(f'[Warning] Missing class directory: {class_dir}')
            continue

        for file_path in sorted(class_dir.rglob('*')):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTS:
                yield file_path, class_name


def gen_dataset_from_folder(root: Path, augment: bool = False):
    x_audio_out = []
    x_feat_out = []
    y_out = []
    file_count = 0
    segment_count = 0

    print(f'\nLoading data from: {root}')
    for audio_path, class_name in iter_audio_files(root):
        try:
            raw_data = load_audio(audio_path, sr=SR)

            if len(raw_data) < SAMPLE_SIZE:
                print(f'[Skip] Too short: {audio_path}')
                continue

            stride = SAMPLE_SIZE
            aug_times = 2 if augment else 1

            for start in range(0, len(raw_data) - SAMPLE_SIZE + 1, stride):
                split_audio = raw_data[start:start + SAMPLE_SIZE]

                for aug_idx in range(aug_times):
                    if augment and aug_idx > 0:
                        std = np.random.uniform(0.001, 0.03)
                        proc_audio = add_noise(split_audio, std=std)
                    else:
                        proc_audio = split_audio

                    feature_combine = extract_feature_vector(proc_audio, sr=SR)
                    x_audio_out.append(proc_audio.astype(np.float32))
                    x_feat_out.append(feature_combine)
                    y_out.append(CLASS_TO_INDEX[class_name])
                    segment_count += 1

            file_count += 1

        except Exception as e:
            print(f'[Error] {audio_path}: {e}')

    if len(x_audio_out) == 0:
        raise RuntimeError(f'No valid audio segments found under {root}')

    x_audio = np.array(x_audio_out, dtype=np.float32)[..., np.newaxis]
    x_feat = np.array(x_feat_out, dtype=np.float32)[..., np.newaxis]
    y_idx = np.array(y_out, dtype=np.int32)
    y_onehot = keras.utils.to_categorical(y_idx, num_classes=len(CLASS_NAMES))

    print(f'Loaded files: {file_count}')
    print(f'Generated segments: {segment_count}')
    print(f'x_audio shape: {x_audio.shape}')
    print(f'x_feat shape:  {x_feat.shape}')
    print(f'y shape:       {y_onehot.shape}')

    return x_audio, x_feat, y_onehot, y_idx


# =========================
# Model
# =========================
def build_multiscale_conv(time_window: int, out_num: int):
    """
    修改分支 1: 加入多尺度 CNN (Multi-scale CNN)
    """
    input_layer_1d = keras.Input(shape=(time_window, 1), name='time_series_inputs')
    x = layers.Reshape((10, 120, 160, 1))(input_layer_1d)

    # Scale 1: 3x3 Convolution
    conv_scale_1 = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    conv_scale_1 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(conv_scale_1)

    # Scale 2: 5x5 Convolution (捕捉更大感受野的特徵)
    conv_scale_2 = layers.TimeDistributed(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))(x)
    conv_scale_2 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(conv_scale_2)

    # Concatenate scales
    concat_scales = layers.Concatenate(name='concat_multiscale')([conv_scale_1, conv_scale_2])

    # 繼續原有的深層萃取
    x_deep = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(concat_scales)
    x_deep = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x_deep)

    x_flat = layers.TimeDistributed(layers.Flatten())(x_deep)
    x_lstm = layers.LSTM(64, return_sequences=False)(x_flat)
    x_drop = layers.Dropout(0.2)(x_lstm)

    output_layer = layers.Dense(out_num, activation='relu', name='multiscale_audio_out')(x_drop)
    return keras.Model(inputs=input_layer_1d, outputs=output_layer, name='audio_branch_multiscale')


def build_feature_branch(feat_num: int = FEATURE_DIM):
    input_layer = keras.Input(shape=(feat_num, 1), name='feat_inputs')
    x = layers.Flatten()(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    output_layer = layers.Dense(32, activation='relu')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer, name='feature_branch')


def build_model():
    # 1. 取得兩個分支的模型
    conv = build_multiscale_conv(time_window=SAMPLE_SIZE, out_num=64)
    feat_branch = build_feature_branch(feat_num=FEATURE_DIM)

    # 2. 將分支 1 與分支 2 串接
    combined = layers.Concatenate(name='concat_branches')([conv.output, feat_branch.output])
    combined_dense = layers.Dense(128, activation='relu', name='fusion_dense')(combined)
    
    # 為了送入 MultiHeadAttention，將維度由 (Batch, 128) 轉為 (Batch, 1, 128)
    seq_features = layers.Reshape((1, 128), name='reshape_for_attention')(combined_dense)

    # 3. 建立 4 個類別專屬的 Self-Attention 子分支
    attention_outputs = []
    for class_name in CLASS_NAMES:
        # 每個分支使用獨立的 MultiHeadAttention 來學習該類別專屬特徵
        attn_layer = layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=32, 
            name=f'attention_{class_name}'
        )(seq_features, seq_features) # Query=seq_features, Value=seq_features
        
        # 攤平回 1D (Batch, 128)
        attn_flat = layers.Flatten(name=f'flatten_{class_name}')(attn_layer)
        
        # 降維萃取該類別的核心特徵
        class_specific_feat = layers.Dense(
            32, 
            activation='relu', 
            name=f'feature_{class_name}'
        )(attn_flat)
        
        attention_outputs.append(class_specific_feat)

    # 4. 將 4 個 Attention 輸出的特徵串接
    final_concat = layers.Concatenate(name='concat_all_attentions')(attention_outputs)
    final_features = layers.Dropout(0.2, name='final_dropout')(final_concat)
    
    # 5. 最終分類層
    classification = layers.Dense(len(CLASS_NAMES), activation='softmax', name='emotion_output')(final_features)

    model = keras.Model(
        inputs=[conv.input, feat_branch.input],
        outputs=classification,
        name='MER_smile_multiscale_attention_model'
    )
    return model


# =========================
# Plot helpers
# =========================
def save_training_curves(history, save_dir: Path):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'acc.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'loss.png', dpi=200)
    plt.close()


def save_confusion_matrix(y_true_idx, y_pred_idx, save_dir: Path):
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=200)
    plt.close()


# =========================
# Save model helpers
# =========================
def save_final_model(model: keras.Model, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = model_dir / 'final_model.keras'
    final_weights_path = model_dir / 'final_model.weights.h5'

    model.save(final_model_path)
    model.save_weights(final_weights_path)

    print(f'\n[Saved] Full model (architecture + weights): {final_model_path}')
    print(f'[Saved] Weights only: {final_weights_path}')


# =========================
# Main
# =========================
def main():
    print('TensorFlow version:', tf.__version__)
    print('GPU available:', bool(tf.config.list_physical_devices('GPU')))

    x_audio_train, x_feat_train, y_train, y_train_idx = gen_dataset_from_folder(TRAIN_ROOT, augment=True)
    x_audio_test, x_feat_test, y_test, y_test_idx = gen_dataset_from_folder(TEST_ROOT, augment=False)

    print('\nClass distribution (train):', np.sum(y_train, axis=0).astype(int), '->', CLASS_NAMES)
    print('Class distribution (test): ', np.sum(y_test, axis=0).astype(int), '->', CLASS_NAMES)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print('\n================ Model Summary ================')
    # 1. 依然在終端機印出一次方便查看
    model.summary()
    
    # 2. 將 Model Summary 寫入到 ModelSummy.txt
    summary_path = RESULTS_DIR / 'ModelSummy.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        # 寫入自訂標題 (可選)
        f.write("================ MER_smile Model Summary ================\n\n")
        # 透過 print_fn 將 summary 每行輸出寫入檔案
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    print(f'[Saved] Model summary saved to: {summary_path}')

    try:
        tf.keras.utils.plot_model(
            model,
            to_file=str(RESULTS_DIR / 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=120
        )
        print(f'Model architecture image saved to: {RESULTS_DIR / "model_architecture.png"}')
    except Exception as e:
        print(f'[Warning] Could not save model architecture image: {e}')

    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_DIR / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        [x_audio_train, x_feat_train],
        y_train,
        validation_data=([x_audio_test, x_feat_test], y_test),
        epochs=50,
        batch_size=4,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    print('\n================ Final Evaluation on Test Set ================')
    test_loss, test_acc = model.evaluate([x_audio_test, x_feat_test], y_test, verbose=1)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

    pred_prob = model.predict([x_audio_test, x_feat_test], verbose=1)
    pred_idx = np.argmax(pred_prob, axis=1)

    print('\n================ Classification Report ================')
    print(classification_report(y_test_idx, pred_idx, target_names=CLASS_NAMES, digits=4))

    save_training_curves(history, RESULTS_DIR)
    save_confusion_matrix(y_test_idx, pred_idx, RESULTS_DIR)
    save_final_model(model, MODEL_DIR)

    print('\nSaved result files:')
    print(RESULTS_DIR / 'loss.png')
    print(RESULTS_DIR / 'acc.png')
    print(RESULTS_DIR / 'confusion_matrix.png')
    if (RESULTS_DIR / 'model_architecture.png').exists():
        print(RESULTS_DIR / 'model_architecture.png')

    print('\nSaved model files:')
    print(MODEL_DIR / 'best_model.keras')
    print(MODEL_DIR / 'final_model.keras')
    print(MODEL_DIR / 'final_model.weights.h5')


if __name__ == '__main__':
    main()