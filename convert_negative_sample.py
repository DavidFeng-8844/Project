import json
from pathlib import Path
from PIL import Image

coco_json = Path("./Datasets/my_coco_dataset/dataset.json")
image_dir = Path("./Datasets/original")  # 你的图片所在目录

with open(coco_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# 已有图片名集合（用basename对齐）
existing = {Path(img["file_name"]).name for img in data["images"]}

# 扫描目录下所有图片
img_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

# 准备新的起始 id
next_id = (max([img["id"] for img in data["images"]] or [0]) + 1)

added = 0
for p in img_paths:
    if p.name in existing:
        continue
    with Image.open(p) as im:
        width, height = im.size
    data["images"].append({
        "id": next_id,
        "file_name": p.name,  # 我们的代码会用 basename 去找文件，保持这样即可
        "width": width,
        "height": height
    })
    next_id += 1
    added += 1

if added:
    with open(coco_json, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Added {added} negative images to COCO 'images'.")
else:
    print("No new negatives found.")