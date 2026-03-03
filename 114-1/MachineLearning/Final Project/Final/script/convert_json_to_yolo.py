import json
import os
from pathlib import Path
import shutil

def convert_coco_to_yolo(json_path, output_dir):
    """
    Converts COCO JSON annotations to YOLO TXT format.
    """
    print(f"Processing {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map image IDs to image info
    images = {img['id']: img for img in data['images']}
    
    # Map category IDs to YOLO class indices (assuming 0-indexed contiguous)
    # If your classes are not 0-indexed or contiguous, you might need a mapping.
    # Here we assume category_id 1 is class 0, etc., or just use the category_id if it starts at 0.
    # Let's check categories first.
    categories = data.get('categories', [])
    # Create a mapping from COCO category ID to YOLO class index (0, 1, 2...)
    # Sort by id to ensure consistent ordering
    categories.sort(key=lambda x: x['id'])
    cat_id_to_yolo_idx = {cat['id']: i for i, cat in enumerate(categories)}
    
    print(f"Categories mapping: {cat_id_to_yolo_idx}")

    # Group annotations by image_id
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    count = 0
    for img_id, img_info in images.items():
        file_name = img_info['file_name']
        # YOLO label filename corresponds to image filename
        label_name = Path(file_name).stem + ".txt"
        label_path = output_dir / label_name
        
        img_w = img_info['width']
        img_h = img_info['height']
        
        anns = img_annotations.get(img_id, [])
        
        yolo_lines = []
        for ann in anns:
            bbox = ann['bbox'] # [x, y, w, h]
            cat_id = ann['category_id']
            
            # Convert to YOLO format: class x_center y_center width height (normalized)
            x, y, w, h = bbox
            
            # Center coordinates
            x_center = x + w / 2.0
            y_center = y + h / 2.0
            
            # Normalize
            x_center /= img_w
            y_center /= img_h
            w /= img_w
            h /= img_h
            
            # Clip to [0, 1] just in case
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            class_idx = cat_id_to_yolo_idx.get(cat_id)
            if class_idx is None:
                print(f"Warning: Category ID {cat_id} not found in categories list. Skipping.")
                continue
                
            yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            count += 1
            
    print(f"Generated {count} label files in {output_dir}")

def main():
    # Define paths
    project_root = Path("/home/liang/workdir/NCKU-ML-25Fall/Final_Project")
    dataset_tiled_dir = project_root / "dataset_tiled_640"
    
    # Train set
    train_json = dataset_tiled_dir / "labels" / "instances_train_tiled.json"
    train_labels_dir = dataset_tiled_dir / "labels" / "train"
    
    if train_json.exists():
        convert_coco_to_yolo(train_json, train_labels_dir)
    else:
        print(f"Error: {train_json} does not exist.")

    # Val set
    val_json = dataset_tiled_dir / "labels" / "instances_val_tiled.json"
    val_labels_dir = dataset_tiled_dir / "labels" / "val"
    
    if val_json.exists():
        convert_coco_to_yolo(val_json, val_labels_dir)
    else:
        print(f"Error: {val_json} does not exist.")

if __name__ == "__main__":
    main()
