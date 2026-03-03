
from pathlib import Path
import os
import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1] # Assumes script is in Final_Project/script/

MODEL_PATH = ROOT / "runs" / "train" / "yolo11l.pt_1024px_b0.7_e10002" / "weights" / "best.pt" # Path to your YOLO model weights
# Directory containing the whole slide images to be analyzed (e.g., validation set)
# Now pointing to the extracted WSI
INPUT_WSIS_ROOT = ROOT / "original_wsis"

# Output directory for results (e.g., CSV report)
OUTPUT_DIR = ROOT / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SAHI parameters for slicing inference (should match training/previous inference)
SLICE_HEIGHT = 1024
SLICE_WIDTH = 1024
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2
CONFIDENCE_THRESHOLD = 0.25 # Confidence threshold for YOLO detections

# Classification thresholds (these are placeholders, adjust as needed)
# Count of abnormal detections
LOW_COUNT_THRESHOLD = 2
HIGH_COUNT_THRESHOLD = 5

# Proportion of abnormal area (e.g., 0.01 means 1%)
LOW_AREA_PROPORTION_THRESHOLD = 0.001 # 0.1%
HIGH_AREA_PROPORTION_THRESHOLD = 0.01  # 1%

# --- Debug/Limited Run Configuration ---
LIMIT_IMAGES_TO_PROCESS = 20 # Set to None to process all images, or an integer for a limited run

# --- Main Analysis Function ---
def analyze_slide_detections(image_path: Path, detection_model, min_conf: float = 0.25):
    """
    Analyzes detections on a single whole slide image using SAHI.

    Args:
        image_path (Path): Path to the whole slide image.
        detection_model: SAHI AutoDetectionModel instance.
        min_conf (float): Minimum confidence threshold for detections to be considered.

    Returns:
        tuple: (total_detections, abnormal_area_proportion)
               Returns (0, 0) if image cannot be read or no detections.
    """
    try:
        # Get image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return 0, 0
        
        image_height, image_width, _ = img.shape
        total_image_area = image_width * image_height

        print(f"Analyzing {image_path.name} (Resolution: {image_width}x{image_height})...")

        # Perform sliced prediction
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            verbose=0 # Suppress SAHI verbose output
        )

        total_detections = 0
        total_abnormal_area_pixels = 0

        # Process each detected object
        for obj_prediction in result.object_prediction_list:
            if obj_prediction.score.value >= min_conf:
                total_detections += 1
                
                # Bounding box is in [x1, y1, x2, y2] format
                x1, y1, x2, y2 = obj_prediction.bbox.to_xyxy()
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                total_abnormal_area_pixels += (bbox_width * bbox_height)
        
        abnormal_area_proportion = total_abnormal_area_pixels / total_image_area if total_image_area > 0 else 0

        return total_detections, abnormal_area_proportion

    except Exception as e:
        print(f"An error occurred while analyzing {image_path.name}: {e}")
        return 0, 0

# --- Script Execution ---
if __name__ == "__main__":
    # 1. Load the YOLO model with SAHI
    print(f"Loading YOLO model from {MODEL_PATH}...")
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', # Assuming YOLOv8 based on yolo11n.pt and ultralytics import
            model_path=str(MODEL_PATH),
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device="cuda:0" # Use GPU if available, change to "cpu" if not
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Please ensure ultralytics and sahi are installed and the model path is correct.")
        print("Attempting to load as generic YOLO model, might not work without correct type.")
        # Fallback to generic YOLO if specific type fails, might not work without correct type
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolo',
            model_path=str(MODEL_PATH),
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device="cuda:0"
        )
        print("Loaded as generic YOLO model (may require manual specification of model_type if errors occur).")


    results = []
    
    # Collect all image files from both negative and positive subdirectories
    image_files = []
    for sub_dir in ["negative", "positive"]:
        current_dir = INPUT_WSIS_ROOT / sub_dir
        image_files.extend(list(current_dir.glob("*.png")))
        image_files.extend(list(current_dir.glob("*.jpg"))) # Include JPGs if present

    if not image_files:
        print(f"No images found in {INPUT_WSIS_ROOT}/negative or {INPUT_WSIS_ROOT}/positive. Please check the path and content.")
    else:
        print(f"Found {len(image_files)} images for analysis.")
        processed_count = 0
        for image_path in image_files:
            if LIMIT_IMAGES_TO_PROCESS is not None and processed_count >= LIMIT_IMAGES_TO_PROCESS:
                print(f"Reached image processing limit ({LIMIT_IMAGES_TO_PROCESS}). Stopping.")
                break

            total_detections, abnormal_area_proportion = analyze_slide_detections(
                image_path, detection_model, CONFIDENCE_THRESHOLD
            )
            processed_count += 1


            # Apply classification logic
            classification = "UNKNOWN"
            if total_detections == 0:
                classification = "LOW_RISK"
            elif total_detections < LOW_COUNT_THRESHOLD and abnormal_area_proportion < LOW_AREA_PROPORTION_THRESHOLD:
                classification = "LOW_RISK"
            elif total_detections >= HIGH_COUNT_THRESHOLD or abnormal_area_proportion >= HIGH_AREA_PROPORTION_THRESHOLD:
                classification = "HIGH_RISK"
            else:
                classification = "MEDIUM_RISK" # Or "UNCERTAIN_REVIEW"

            results.append({
                "slide_name": image_path.name,
                "total_detections": total_detections,
                "abnormal_area_proportion": f"{abnormal_area_proportion:.4f}", # Format for readability
                "classification": classification
            })
    
    # --- Save Results to CSV ---
    output_csv_path = OUTPUT_DIR / "analysis_summary.csv"
    with open(output_csv_path, 'w') as f:
        f.write("slide_name,total_detections,abnormal_area_proportion,classification\n")
        for row in results:
            f.write(f"{row['slide_name']},{row['total_detections']},{row['abnormal_area_proportion']},{row['classification']}\n")

    print(f"\nAnalysis complete. Results saved to {output_csv_path}")
    print("\n--- Summary ---")
    for r in results:
        print(f"Slide: {r['slide_name']}, Detections: {r['total_detections']}, Abnormal Area: {float(r['abnormal_area_proportion'])*100:.2f}%, Classification: {r['classification']}")

    print("\nNext Steps:")
    print(f"1. Review '{output_csv_path}' for the classified slides.")
    print("2. For 'MEDIUM_RISK' slides, manually review their corresponding heatmaps with a medical expert.")
    print("3. Adjust LOW_COUNT_THRESHOLD, HIGH_COUNT_THRESHOLD, LOW_AREA_PROPORTION_THRESHOLD, HIGH_AREA_PROPORTION_THRESHOLD in the script based on expert feedback.")
