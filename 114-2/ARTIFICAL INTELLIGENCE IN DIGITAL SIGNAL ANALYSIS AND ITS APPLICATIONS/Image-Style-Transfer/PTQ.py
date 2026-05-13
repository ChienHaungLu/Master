# =====================================================
# TensorRT PTQ (Post-Training Quantization)
# AdaIN INT8 Engine Builder for Jetson Xavier NX
#
# 功能:
# 1. 使用 TensorRT INT8 Calibration 建立 INT8 engine
# 2. 顯示量化前後模型大小
# 3. 產生 calibration cache
#
# 執行方式:
# python PTQ.py
#
# 輸出:
#   adain_jetson_int8.engine
#   adain_int8_calibration.cache
# =====================================================

import os
import glob
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# =====================================================
# 1. 路徑設定
# =====================================================

# 原始 ONNX 模型
ONNX_PATH = "adain_jetson.onnx"

# 原本 FP32 TensorRT engine (可選)
FP32_ENGINE_PATH = "adain_jetson.engine"

# 輸出的 INT8 engine
INT8_ENGINE_PATH = "adain_jetson_int8.engine"

# Calibration cache
CALIB_CACHE = "adain_int8_calibration.cache"

# Calibration 圖片資料夾
CONTENT_CALIB_DIR = "./calibration/content"
STYLE_CALIB_DIR = "./calibration/style"


# =====================================================
# 2. 基本參數
# =====================================================

INPUT_SIZE = (512, 512)
BATCH_SIZE = 1

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# =====================================================
# 3. 印出檔案大小
# =====================================================

def print_file_size(path, label):
    """
    顯示檔案大小
    """
    if os.path.exists(path):
        size_bytes = os.path.getsize(path)
        size_kb = size_bytes / 1024
        size_mb = size_bytes / (1024 * 1024)

        print(f"{label}")
        print(f"Path : {path}")
        print(f"Size : {size_bytes:,} Bytes")
        print(f"       {size_kb:.2f} KB")
        print(f"       {size_mb:.2f} MB")
        print("----------------------------------")

    else:
        print(f"{label}")
        print(f"{path} not found")
        print("----------------------------------")


# =====================================================
# 4. 前處理
# =====================================================

def load_image(path):
    """
    讀取 calibration image
    Output:
        shape = (1,3,512,512)
        dtype = float32
        range = 0~1
    """
    image = Image.open(path).convert("RGB")

    image = image.resize(INPUT_SIZE)

    arr = np.array(image).astype(np.float32) / 255.0

    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))

    # CHW -> NCHW
    arr = np.expand_dims(arr, axis=0)

    return np.ascontiguousarray(arr, dtype=np.float32)


# =====================================================
# 5. INT8 Calibrator
# =====================================================

class AdaINCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, content_dir, style_dir, cache_file):
        super().__init__()

        self.cache_file = cache_file

        # 取得 calibration image path
        content_paths = sorted(glob.glob(os.path.join(content_dir, "*")))
        style_paths = sorted(glob.glob(os.path.join(style_dir, "*")))

        self.pairs = list(zip(content_paths, style_paths))

        if len(self.pairs) == 0:
            raise RuntimeError(
                "\nNo calibration image pairs found.\n"
                "請確認 calibration/content 與 calibration/style 有圖片。\n"
            )

        self.index = 0

        # input shape
        self.content_shape = (1, 3, 512, 512)
        self.style_shape = (1, 3, 512, 512)

        # 計算 bytes
        self.content_nbytes = np.empty(
            self.content_shape,
            dtype=np.float32
        ).nbytes

        self.style_nbytes = np.empty(
            self.style_shape,
            dtype=np.float32
        ).nbytes

        # 配置 GPU memory
        self.d_content = cuda.mem_alloc(self.content_nbytes)
        self.d_style = cuda.mem_alloc(self.style_nbytes)

        print("\n========== Calibration Dataset ==========")
        print(f"Calibration pairs : {len(self.pairs)}")
        print(f"Content folder    : {content_dir}")
        print(f"Style folder      : {style_dir}")
        print("----------------------------------")

    def get_batch_size(self):
        return BATCH_SIZE

    def get_batch(self, names):

        if self.index >= len(self.pairs):
            return None

        content_path, style_path = self.pairs[self.index]

        # load image
        content = load_image(content_path)
        style = load_image(style_path)

        # copy Host -> Device
        cuda.memcpy_htod(self.d_content, content)
        cuda.memcpy_htod(self.d_style, style)

        self.index += 1

        print(
            f"Calibrating batch "
            f"{self.index}/{len(self.pairs)}"
        )

        # TensorRT bindings
        return [
            int(self.d_content),
            int(self.d_style)
        ]

    def read_calibration_cache(self):

        if os.path.exists(self.cache_file):

            print("\nUsing existing calibration cache.")

            with open(self.cache_file, "rb") as f:
                return f.read()

        return None

    def write_calibration_cache(self, cache):

        print("\nWriting calibration cache...")

        with open(self.cache_file, "wb") as f:
            f.write(cache)


# =====================================================
# 6. 建立 INT8 TensorRT Engine
# =====================================================

def build_int8_engine():

    print("\n========== Building INT8 Engine ==========")

    builder = trt.Builder(TRT_LOGGER)

    # Explicit batch
    network_flags = 1 << int(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    )

    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, TRT_LOGGER)

    # -------------------------------------------------
    # Parse ONNX
    # -------------------------------------------------

    print(f"\nLoading ONNX model: {ONNX_PATH}")

    with open(ONNX_PATH, "rb") as f:

        success = parser.parse(f.read())

        if not success:

            print("\nFailed to parse ONNX model.\n")

            for i in range(parser.num_errors):
                print(parser.get_error(i))

            return None

    print("ONNX model parsed successfully!")

    # -------------------------------------------------
    # Builder config
    # -------------------------------------------------

    config = builder.create_builder_config()

    # workspace size
    config.max_workspace_size = 1 << 30  # 1GB

    # 檢查平台 INT8 支援
    if builder.platform_has_fast_int8:
        print("\nPlatform supports FAST INT8.")
    else:
        print("\nWARNING: Platform may not have FAST INT8 support.")

    # 啟用 INT8
    config.set_flag(trt.BuilderFlag.INT8)

    # -------------------------------------------------
    # Calibration
    # -------------------------------------------------

    calibrator = AdaINCalibrator(
        CONTENT_CALIB_DIR,
        STYLE_CALIB_DIR,
        CALIB_CACHE
    )

    config.int8_calibrator = calibrator

    # -------------------------------------------------
    # Build engine
    # -------------------------------------------------

    print("\nBuilding INT8 TensorRT engine...")
    print("This may take several minutes...\n")

    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError("Failed to build INT8 engine.")

    print("\nINT8 engine build SUCCESS!")

    # -------------------------------------------------
    # Save engine
    # -------------------------------------------------

    with open(INT8_ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())

    print(f"\nINT8 engine saved to:")
    print(INT8_ENGINE_PATH)

    return engine


# =====================================================
# 7. 主程式
# =====================================================

if __name__ == "__main__":

    print("\n=================================================")
    print("TensorRT PTQ (Post-Training Quantization)")
    print("AdaIN INT8 Engine Builder")
    print("=================================================")

    # -------------------------------------------------
    # Before PTQ
    # -------------------------------------------------

    print("\n========== Before PTQ ==========")

    print_file_size(
        ONNX_PATH,
        "Original ONNX Model"
    )

    print_file_size(
        FP32_ENGINE_PATH,
        "FP32 TensorRT Engine"
    )

    # -------------------------------------------------
    # Build INT8 Engine
    # -------------------------------------------------

    build_int8_engine()

    # -------------------------------------------------
    # After PTQ
    # -------------------------------------------------

    print("\n========== After PTQ ==========")

    print_file_size(
        INT8_ENGINE_PATH,
        "INT8 TensorRT Engine"
    )

    print_file_size(
        CALIB_CACHE,
        "Calibration Cache"
    )

    # -------------------------------------------------
    # Compression Ratio
    # -------------------------------------------------

    if os.path.exists(FP32_ENGINE_PATH) and os.path.exists(INT8_ENGINE_PATH):

        fp32_size = os.path.getsize(FP32_ENGINE_PATH)
        int8_size = os.path.getsize(INT8_ENGINE_PATH)

        reduction = (
            (fp32_size - int8_size) / fp32_size
        ) * 100

        print("\n========== Compression Result ==========")

        print(f"FP32 Engine : {fp32_size / (1024*1024):.2f} MB")
        print(f"INT8 Engine : {int8_size / (1024*1024):.2f} MB")
        print(f"Reduction   : {reduction:.2f}%")

        print("----------------------------------")

    print("\nDone!")