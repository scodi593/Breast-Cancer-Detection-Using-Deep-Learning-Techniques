# ===============================
# 1. Imports and Configs
# ===============================
import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# Paths
base_path = "C:/Users/SATHVIKA/Desktop/d/dataset"  # ✅ Update if needed
output_path = "./YOLO3_data"
splits = ['train', 'val', 'test']
class_map = {'benign': 0, 'malignant': 1}

# Create output folders
for split in splits:
    os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', split), exist_ok=True)

# ===============================
# 2. Collect and Shuffle Images
# ===============================
all_data = []
for class_name, class_id in class_map.items():
    class_dir = os.path.join(base_path, class_name)
    if os.path.exists(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.png')):
                all_data.append((os.path.join(class_dir, fname), class_id))

random.shuffle(all_data)
total = len(all_data)
train_split = int(0.7 * total)
val_split = int(0.9 * total)

splitted = {
    'train': all_data[:train_split],
    'val': all_data[train_split:val_split],
    'test': all_data[val_split:]
}

# ===============================
# 3. Preprocessing + Annotation
# ===============================
def preprocess_and_label(image_path, class_id):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, []

    # Resize to 416x416
    gray = cv2.resize(gray, (416, 416))

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Remove pectoral muscle
    mask = np.ones_like(gray, dtype=np.uint8) * 255
    triangle_h, triangle_w = 130, 130
    left_triangle = np.array([[0, 0], [triangle_w, 0], [0, triangle_h]], np.int32)
    right_triangle = np.array([[416, 0], [416 - triangle_w, 0], [416, triangle_h]], np.int32)

    if np.mean(gray[:, :50]) > np.mean(gray[:, -50:]):
        cv2.fillPoly(mask, [left_triangle], 0)
    else:
        cv2.fillPoly(mask, [right_triangle], 0)

    gray = cv2.bitwise_and(gray, mask)

    # Crop breast tissue
    _, tissue_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        gray = gray[y:y+h, x:x+w]
    gray = cv2.resize(gray, (416, 416))

    # Normalize
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Lesion detection (contour-based)
    lesion_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < 8000:
            x, y, w, h = cv2.boundingRect(cnt)
            x_center = (x + w / 2) / 416
            y_center = (y + h / 2) / 416
            width = w / 416
            height = h / 416
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # ✅ Return grayscale image (no RGB conversion)
    return gray, annotations

# ===============================
# 4. Process and Save
# ===============================
for split in splits:
    print(f"\n🔄 Processing {split} set...")
    for image_path, class_id in tqdm(splitted[split]):
        try:
            image, labels = preprocess_and_label(image_path, class_id)
            if image is None:
                continue

            fname = os.path.basename(image_path)
            img_save_path = os.path.join(output_path, 'images', split, fname)
            cv2.imwrite(img_save_path, image)  # ✅ Save grayscale

            label_name = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(output_path, 'labels', split, label_name)
            with open(label_path, 'w') as f:
                f.write("\n".join(labels) if labels else "")

        except Exception as e:
            print(f"[❌] Error: {image_path} — {e}")

print("\n✅ Preprocessing complete. YOLO-formatted grayscale dataset is ready.")
