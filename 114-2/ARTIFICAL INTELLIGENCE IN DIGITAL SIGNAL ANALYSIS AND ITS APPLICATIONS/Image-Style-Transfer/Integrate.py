# =====================================================
# Image Mode:
# CPU:
# python Integrate.py --mode image --content ./content/1.jpg --style ./style/1.jpg --device cpu
#
# ONNX Runtime GPU:
# python Integrate.py --mode image --content ./content/1.jpg --style ./style/1.jpg --device gpu
#
# TensorRT GPU: (只限在Xaiver使用)
# python Integrate.py --mode image --content ./content/1.jpg --style ./style/1.jpg --device trt --engine adain_jetson_int8.engine
#
# Camera Mode:
# CPU: (camera parameter 0 or 1)
# python Integrate.py --mode camera --style ./style/1.jpg --device cpu --camera 0
#
# ONNX Runtime GPU: (camera parameter 0 or 1)
# python Integrate.py --mode camera --style ./style/1.jpg --device gpu --camera 0
#
# TensorRT GPU: (camera parameter 0 or 1)(只限在Xaiver使用)
# python Integrate.py --mode camera --style ./style/1.jpg --device trt --engine adain_jetson_int8.engine --camera 0
# =====================================================

import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort


# =====================================================
# 0. 檢查 Camera
# =====================================================
def list_cameras(max_tested=2):
    print("\nSearching cameras...")

    available = []

    for i in range(max_tested):
        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                print(f"[OK] Camera index: {i}")
                available.append(i)

            cap.release()

    if len(available) == 0:
        print("No camera found.")

    return available


# =====================================================
# 1. 前處理
# =====================================================
def preprocess_pil(img, input_size=512, resize=True):
    """
    PIL RGB image -> model input
    Output shape: (1, 3, H, W)
    dtype: float32
    range: 0~1
    """
    img = img.convert("RGB")

    if resize:
        img = img.resize((input_size, input_size), Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))   # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)    # CHW -> NCHW

    return np.ascontiguousarray(arr, dtype=np.float32)


# =====================================================
# 2. 後處理
# =====================================================
def postprocess_to_rgb(output):
    """
    Model output -> RGB uint8 image
    Input shape: (1, 3, H, W)
    Output shape: (H, W, 3)
    """
    output = np.squeeze(output, axis=0)
    output = np.transpose(output, (1, 2, 0))
    output = np.clip(output, 0.0, 1.0)
    output = (output * 255).astype(np.uint8)
    return output


def postprocess_to_bgr(output, target_w=None, target_h=None):
    """
    Model output -> OpenCV BGR image
    """
    output_rgb = postprocess_to_rgb(output)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    if target_w is not None and target_h is not None:
        output_bgr = cv2.resize(
            output_bgr,
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC
        )

    return output_bgr


# =====================================================
# 3. 建立 ONNX Runtime Session
# =====================================================
def create_session(onnx_path, device):
    """
    cpu: CPUExecutionProvider
    gpu: CUDAExecutionProvider
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:
            print(f"Warning: ort.preload_dlls() failed: {e}")

    available_providers = ort.get_available_providers()

    print("\nAvailable ONNX Runtime Providers:")
    print(available_providers)

    if device == "gpu":
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "\n你指定了 --device gpu，但目前沒有 CUDAExecutionProvider。\n"
                "在 Jetson Xavier NX 上，建議使用 --device trt 走 TensorRT engine。\n"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    else:
        providers = ["CPUExecutionProvider"]

    print("\nLoading ONNX model...")

    session = ort.InferenceSession(
        onnx_path,
        providers=providers
    )

    actual_provider = session.get_providers()[0]

    print("ONNX model loaded successfully!")
    print(f"Using Provider: {actual_provider}")

    if actual_provider == "CUDAExecutionProvider":
        print("\n===================================")
        print("Inference Device : ONNX Runtime GPU")
        print("CUDAExecutionProvider ENABLED")
        print("===================================\n")

    else:
        print("\n===================================")
        print("Inference Device : CPU")
        print("CPUExecutionProvider ENABLED")
        print("===================================\n")

    print("\n========== Model Inputs ==========")

    for inp in session.get_inputs():
        print(f"Name  : {inp.name}")
        print(f"Shape : {inp.shape}")
        print(f"Type  : {inp.type}")
        print("----------------------------------")

    print("\n========== Model Outputs ==========")

    for out in session.get_outputs():
        print(f"Name  : {out.name}")
        print(f"Shape : {out.shape}")
        print(f"Type  : {out.type}")
        print("----------------------------------")

    output_names = [out.name for out in session.get_outputs()]

    return session, output_names


# =====================================================
# 4. TensorRT Runner
#    讓 TensorRT 也支援 session.run(...)
# =====================================================
class TensorRTRunner:
    def __init__(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # 初始化 CUDA context

        self.trt = trt
        self.cuda = cuda

        print("\nLoading TensorRT engine...")
        print(f"Engine path: {engine_path}")
        print("TensorRT version:", trt.__version__)

        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()

        print("TensorRT engine loaded successfully!")

        print("\n===================================")
        print("Inference Device : TensorRT GPU")
        print("TensorRT Engine ENABLED")
        print("===================================\n")

        print("\n========== TensorRT Bindings ==========")
        for i in range(self.engine.num_bindings):
            print(f"Index : {i}")
            print(f"Name  : {self.engine.get_binding_name(i)}")
            print(f"Shape : {self.engine.get_binding_shape(i)}")
            print(f"Dtype : {self.engine.get_binding_dtype(i)}")
            print(f"Input : {self.engine.binding_is_input(i)}")
            print("----------------------------------")

        self.content_idx = self.engine.get_binding_index("content")
        self.style_idx = self.engine.get_binding_index("style")
        self.output_idx = self.engine.get_binding_index("output")

        if self.content_idx < 0:
            raise RuntimeError("TensorRT engine input 'content' not found.")

        if self.style_idx < 0:
            raise RuntimeError("TensorRT engine input 'style' not found.")

        if self.output_idx < 0:
            raise RuntimeError("TensorRT engine output 'output' not found.")

    def run(self, output_names, inputs):
        """
        模擬 ONNX Runtime 的 session.run()
        output_names 參數保留，但 TensorRT 不使用。
        inputs 格式:
        {
            "content": content_input,
            "style": style_input
        }
        """
        content = np.ascontiguousarray(inputs["content"], dtype=np.float32)
        style = np.ascontiguousarray(inputs["style"], dtype=np.float32)

        # 若 engine 是 dynamic shape，這裡設定 input shape
        self.context.set_binding_shape(self.content_idx, content.shape)
        self.context.set_binding_shape(self.style_idx, style.shape)

        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all TensorRT binding shapes are specified.")

        output_shape = tuple(self.context.get_binding_shape(self.output_idx))
        output = np.empty(output_shape, dtype=np.float32)

        # 配置 GPU memory
        d_content = self.cuda.mem_alloc(content.nbytes)
        d_style = self.cuda.mem_alloc(style.nbytes)
        d_output = self.cuda.mem_alloc(output.nbytes)

        bindings = [None] * self.engine.num_bindings
        bindings[self.content_idx] = int(d_content)
        bindings[self.style_idx] = int(d_style)
        bindings[self.output_idx] = int(d_output)

        stream = self.cuda.Stream()

        # Host -> Device
        self.cuda.memcpy_htod_async(d_content, content, stream)
        self.cuda.memcpy_htod_async(d_style, style, stream)

        # TensorRT inference
        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=stream.handle
        )

        # Device -> Host
        self.cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        return [output]


# =====================================================
# 5. 單張圖片模式
# =====================================================
def run_image_mode(args, session, output_names):
    if not args.content:
        raise ValueError("--mode image 需要指定 --content")

    if not os.path.exists(args.content):
        raise FileNotFoundError(f"Content image not found: {args.content}")

    if not os.path.exists(args.style):
        raise FileNotFoundError(f"Style image not found: {args.style}")

    print("\nLoading images...")

    content_img = Image.open(args.content).convert("RGB")
    style_img = Image.open(args.style).convert("RGB")

    content_input = preprocess_pil(
        content_img,
        input_size=args.input_size,
        resize=not args.no_resize
    )

    style_input = preprocess_pil(
        style_img,
        input_size=args.input_size,
        resize=not args.no_resize
    )

    print(f"Content Image : {args.content}")
    print(f"Style Image   : {args.style}")
    print("Content Shape :", content_input.shape)
    print("Style Shape   :", style_input.shape)

    print("\nRunning warm-up...")

    for _ in range(args.warmup):
        _ = session.run(
            output_names,
            {
                "content": content_input,
                "style": style_input
            }
        )

    print("\nRunning benchmark...")

    times = []
    outputs = None

    for i in range(args.repeat):
        start_time = time.time()

        outputs = session.run(
            output_names,
            {
                "content": content_input,
                "style": style_input
            }
        )

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    fps = 1.0 / max(avg_time, 1e-6)

    print("\n========== Performance ==========")
    print(f"Average Inference Time : {avg_time:.4f} seconds")
    print(f"Average FPS            : {fps:.2f}")
    print(f"Min Inference Time     : {min(times):.4f} seconds")
    print(f"Max Inference Time     : {max(times):.4f} seconds")

    result_rgb = postprocess_to_rgb(outputs[0])

    output_dir = os.path.dirname(args.output)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(result_rgb).save(args.output)

    print("\nStylized image saved successfully!")
    print(f"Output Path : {args.output}")


# =====================================================
# 6. Camera 即時模式
# =====================================================
def run_camera_mode(args, session, output_names):
    if not os.path.exists(args.style):
        raise FileNotFoundError(f"Style image not found: {args.style}")

    style_img = Image.open(args.style).convert("RGB")
    style_input = preprocess_pil(
        style_img,
        input_size=args.input_size,
        resize=True
    )

    cap = cv2.VideoCapture(args.camera)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera. Try --camera 1 or check camera permission."
        )

    writer = None

    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.save_video,
            fourcc,
            args.video_fps,
            (args.width, args.height)
        )

    print("\nCamera mode started.")
    print("Press q to quit.")

    prev_time = time.time()

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            print("Cannot read frame from camera.")
            break

        original_h, original_w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        content_img = Image.fromarray(frame_rgb)

        content_input = preprocess_pil(
            content_img,
            input_size=args.input_size,
            resize=True
        )

        outputs = session.run(
            output_names,
            {
                "content": content_input,
                "style": style_input
            }
        )

        output_bgr = postprocess_to_bgr(
            outputs[0],
            target_w=original_w,
            target_h=original_h
        )

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            output_bgr,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("AdaIN Style Transfer", output_bgr)

        if writer is not None:
            frame_to_write = cv2.resize(
                output_bgr,
                (args.width, args.height)
            )
            writer.write(frame_to_write)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            print("Quit camera mode.")
            break

        if cv2.getWindowProperty("AdaIN Style Transfer", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


# =====================================================
# 7. Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["image", "camera"],
        help="image: single image style transfer, camera: real-time camera style transfer"
    )

    parser.add_argument(
        "--content",
        type=str,
        default=None,
        help="Content image path, only used in image mode"
    )

    parser.add_argument(
        "--style",
        type=str,
        required=True,
        help="Style image path"
    )

    parser.add_argument(
        "--onnx",
        type=str,
        default="adain_jetson.onnx",
        help="ONNX model path"
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="adain_jetson.engine",
        help="TensorRT engine path, only used when --device trt"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output/output.jpg",
        help="Output image path, only used in image mode"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "trt"],
        help="cpu = ONNX Runtime CPU, gpu = ONNX Runtime CUDA, trt = TensorRT engine"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Input size for fixed-size ONNX/TensorRT model"
    )

    parser.add_argument(
        "--no_resize",
        action="store_true",
        help="Do not resize image input. Only use this if your model supports dynamic H/W."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, only used in camera mode"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height"
    )

    parser.add_argument(
        "--save_video",
        type=str,
        default="",
        help="Save camera output video path. Empty string means not saving."
    )

    parser.add_argument(
        "--video_fps",
        type=float,
        default=20.0,
        help="Saved video FPS"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up runs before benchmark in image mode"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Benchmark runs in image mode"
    )

    args = parser.parse_args()

    # 可選：列出相機
    list_cameras(max_tested=2)

    if args.device == "trt":
        session = TensorRTRunner(args.engine)
        output_names = None
    else:
        session, output_names = create_session(args.onnx, args.device)

    if args.mode == "image":
        run_image_mode(args, session, output_names)

    elif args.mode == "camera":
        run_camera_mode(args, session, output_names)

    print("\nDone!")


if __name__ == "__main__":
    main()
