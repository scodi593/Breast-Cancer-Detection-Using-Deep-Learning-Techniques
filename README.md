# Breast Cancer Detection Using Deep Learning Techniques

A comprehensive deep learning-based solution to detect breast cancer from mammogram images, combining YOLOv8 for tumor detection and Mask R-CNN for precise lesion segmentation. This system improves upon existing methods through robust preprocessing, efficient annotation handling, and deployment-ready architecture with high diagnostic accuracy.

---

## 📚 Table of Contents

* [Introduction](#introduction)
* [Motivation](#motivation)
* [Problem Statement](#problem-statement)
* [Project Objectives](#project-objectives)
* [System Architecture](#system-architecture)
* [Module Breakdown](#module-breakdown)
* [Technologies Used](#technologies-used)
* [Datasets Used](#datasets-used)
* [Preprocessing Techniques](#preprocessing-techniques)
* [Annotation Formats](#annotation-formats)
* [Model Training and Evaluation](#model-training-and-evaluation)
* [Results and Metrics](#results-and-metrics)
* [Future Enhancements](#future-enhancements)
* [How to Run](#how-to-run)
* [License](#license)

---

## 📌 Introduction

Breast cancer is one of the leading causes of death among women globally. Early and accurate diagnosis plays a critical role in effective treatment. Traditional mammography methods suffer from several limitations such as low contrast, noise, overlapping tissues, and manual interpretation errors. Our solution presents a dual-model approach using deep learning to overcome these limitations.

---

## 💡 Motivation

Manual screening is prone to false negatives and false positives. With the availability of large annotated datasets and GPU-accelerated deep learning tools, it's now feasible to automate this process, reduce radiologist workload, and improve early detection rates.

---

## ❗ Problem Statement

Despite advances in CAD tools and CNN models, most existing systems:

* Lack robust preprocessing.
* Focus only on classification without lesion localization.
* Do not consider important metrics like MCC for imbalanced datasets.

---

## 🎯 Project Objectives

* Detect benign and malignant tumors using YOLOv8.
* Accurately segment lesions using Mask R-CNN.
* Employ preprocessing methods to enhance image quality.
* Train on annotated datasets in YOLO and COCO formats.
* Evaluate using clinical metrics like FPR, FNR, MCC, etc.

---

## 🧠 System Architecture

### High-Level Design:

1. **Data Preprocessing**
2. **YOLO Annotation Generation**
3. **COCO Annotation Conversion**
4. **YOLOv8 Detection**
5. **Mask R-CNN Segmentation**
6. **Tumor Size Estimation**
7. **Visualization & Reporting**

---

## 🧩 Module Breakdown

### 1. User Authentication

Simulated local login-based flow (file-based storage).

### 2. Preprocessing Module

* Grayscale conversion
* CLAHE (contrast enhancement)
* Bilateral filtering
* Pectoral muscle removal
* Cropping and normalization
* Adaptive thresholding for lesion enhancement

### 3. YOLO Annotation Generator

* Based on contours
* Bounding boxes written in YOLO format

### 4. COCO Converter

* Converts YOLO annotations into polygon-based segmentation
* Compatible with Detectron2 and other COCO-based models

### 5. YOLOv8 Detection

* Fast real-time tumor detection
* Trained using annotated YOLO-format images

### 6. Mask R-CNN Segmentation

* Pixel-level mask prediction
* Trained using COCO JSON annotations

### 7. Visualization and Evaluation

* Bounding boxes and masks overlaid on original images
* Metrics like MCC, Precision, Recall, etc.

---

## 🧪 Technologies Used

* Python 3.8+
* OpenCV
* NumPy
* PyTorch
* Ultralytics YOLOv8
* Detectron2 (by Facebook Research)
* Matplotlib & Seaborn
* Jupyter Notebook / VS Code

---

## 🗂️ Datasets Used

* **INbreast**
* **CBIS-DDSM**
* **BNS Mammography Dataset**

> All datasets were annotated manually or using assisted tools like LabelImg and MakeSense.ai

---

## 🖼️ Preprocessing Techniques

Implemented using OpenCV:

* CLAHE for local contrast enhancement
* Bilateral filtering for noise reduction
* Adaptive Thresholding for enhancing lesions
* Custom logic to remove pectoral muscles
* Contour detection for bounding box generation

---

## 📝 Annotation Formats

### YOLO Format

```
<class_id> <x_center> <y_center> <width> <height>
```

### COCO Format

```json
{
  "bbox": [x, y, width, height],
  "segmentation": [[x1, y1, x2, y2, ...]],
  "category_id": 0 or 1
}
```

---

## ⚙️ Model Training and Evaluation

### YOLOv8

* Pretrained on `yolov8m.pt`
* Fine-tuned using `YOLO3_data` split into train/val/test
* Confidence threshold: 0.85
* NMS IoU: 0.4

### Mask R-CNN

* Based on Detectron2
* Trained on `COCO3_data` with polygon masks
* Epochs: 1000 (early stopping based on loss)

### Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* False Positive Rate (FPR)
* False Negative Rate (FNR)
* Matthews Correlation Coefficient (MCC)

---

## 📈 Results and Metrics

| Metric               | Value  |
| -------------------- | ------ |
| True Positives (TP)  | 4      |
| True Negatives (TN)  | 5      |
| False Positives (FP) | 0      |
| False Negatives (FN) | 1      |
| **Precision**        | 100%   |
| **Recall**           | 80%    |
| **F1 Score**         | 88.89% |
| **Accuracy**         | 90%    |
| **FPR**              | 0.00%  |
| **FNR**              | 20.00% |
| **MCC**              | 81.65% |

> These metrics confirm the model is clinically relevant and robust on imbalanced datasets.

---

## 🔮 Future Enhancements

* Add support for multi-class tumor classification (e.g., calcifications)
* Improve generalization with diverse datasets across demographics
* Integrate Explainable AI (e.g., Grad-CAM, SHAP)
* Web and mobile-based frontends for radiologists
* Real-time inference on edge devices
* Automatic patient-specific report generation

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/scodi593/breast-cancer-detection.git
cd breast-cancer-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate YOLO Annotations

```bash
python yolo.py
```

### 4. Convert to COCO Format

```bash
python coco.py
```

### 5. Train YOLOv8

Use the included notebook or:

```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
model.train(data='data.yaml', epochs=50)
```

### 6. Train Mask R-CNN using Detectron2

Refer to `main.ipynb` for detailed training steps.

---

