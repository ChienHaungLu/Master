import os, json
from PIL import Image
from tqdm import tqdm

TILE = 1024
STRIDE = TILE // 2

RAW_IMG = "data/raw_images"
RAW_JSON = "data/raw_coco/annotation.json"

OUT_IMG = "data/tiles/images"
OUT_LBL = "data/tiles/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

with open(RAW_JSON) as f:
    coco = json.load(f)

images = {x["id"]: x for x in coco["images"]}

# group annotations by image
ann_per_img = {}
for ann in coco["annotations"]:
    ann_per_img.setdefault(ann["image_id"], []).append(ann)

def bbox_intersect(bbox, tile):
    x,y,w,h = bbox
    tx,ty,tw,th = tile
    if x+w < tx or x > tx+tw: return False
    if y+h < ty or y > ty+th: return False
    return True

tile_id = 0

for img_id, info in tqdm(images.items()):
    name = info["file_name"]
    img_path = os.path.join(RAW_IMG, name)

    img = Image.open(img_path)
    W, H = img.size

    for y0 in range(0, H-TILE+1, STRIDE):
        for x0 in range(0, W-TILE+1, STRIDE):

            tile = img.crop((x0, y0, x0+TILE, y0+TILE))

            tile_name = f"{tile_id}.jpg"
            tile.save(os.path.join(OUT_IMG, tile_name))

            label_path = os.path.join(OUT_LBL, tile_name.replace(".jpg", ".txt"))
            f_lbl = open(label_path, "w")

            # process annotations inside this tile
            for ann in ann_per_img.get(img_id, []):
                bb = ann["bbox"]  # xywh

                if not bbox_intersect(bb, (x0, y0, TILE, TILE)):
                    continue

                bx, by, bw, bh = bb
                # convert to tile coords
                bx -= x0
                by -= y0

                xc = (bx + bw/2) / TILE
                yc = (by + bh/2) / TILE
                bw /= TILE
                bh /= TILE

                f_lbl.write(f"0 {xc} {yc} {bw} {bh}\n")  # single class Abnormal

            f_lbl.close()

            tile_id += 1

print("✓ Tile + bounding box remap 完成")
