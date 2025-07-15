import os
import json
import cv2
from tqdm import tqdm

# === Paths ===
yolo_base = "C:/Users/SATHVIKA/Desktop/d/YOLO3_data"
coco_base = "C:/Users/SATHVIKA/Desktop/d/COCO3_data"
splits = ['train', 'val']
class_map = {'benign': 0, 'malignant': 1}
categories = [{'id': v, 'name': k, 'supercategory': 'none'} for k, v in class_map.items()]

# === Make Directories ===
os.makedirs(os.path.join(coco_base, "annotations"), exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(coco_base, split), exist_ok=True)

def convert_yolo_to_coco_with_rect_segmentation(split):
    image_dir = os.path.join(yolo_base, 'images', split)
    label_dir = os.path.join(yolo_base, 'labels', split)

    images = []
    annotations = []
    ann_id = 1
    image_id = 1

    for filename in tqdm(os.listdir(image_dir), desc=f"Processing {split}"):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue
        height, width = img.shape[:2]

        # Copy image to COCO folder
        output_img_path = os.path.join(coco_base, split, filename)
        cv2.imwrite(output_img_path, img)

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip() == "":
                        continue

                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"⚠️ Skipping malformed line in {label_path}: {line.strip()}")
                        continue

                    class_id, x_center, y_center, w, h = map(float, parts)
                    class_id = int(class_id)  # ✅ Fix: convert class_id to integer

                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height

                    x_min = x_center - w / 2
                    y_min = y_center - h / 2

                    bbox = [x_min, y_min, w, h]
                    area = w * h

                    segmentation = [[
                        x_min, y_min,
                        x_min + w, y_min,
                        x_min + w, y_min + h,
                        x_min, y_min + h
                    ]]

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id += 1

        image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

# === Process and Save ===
for split in splits:
    coco_data = convert_yolo_to_coco_with_rect_segmentation(split)
    with open(os.path.join(coco_base, "annotations", f"instances_{split}.json"), 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"✅ Saved COCO segmentation annotations for '{split}'")
