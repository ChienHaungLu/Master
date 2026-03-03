import os
import cv2
import pandas as pd
import numpy as np

def rle_encode(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])  # padding
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[:-1:2]
    return ' '.join(str(x) for x in runs)

# 資料夾和輸出的csv存放路徑
mask_dir = "./test_data"
output_csv = "submission.csv"
num_classes = 16       # 0~15 類別

records = []

for fname in sorted(os.listdir(mask_dir)):
    if not fname.endswith(".png"):
        continue
    img_id = fname
    mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
    row = {"img": img_id}
    for class_id in range(num_classes):
        class_mask = (mask == class_id).astype(np.uint8)
        if class_mask.sum() == 0:
            row[f"class_{class_id}"] = "none"
        else:
            row[f"class_{class_id}"] = rle_encode(class_mask)
    records.append(row)

# 存成 CSV
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
