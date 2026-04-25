# AI Inspection SaaS Platform for Power Grids ⚡
> **基于多租户 SaaS 架构的电网智能巡检系统**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-yellow.svg)
![Flask](https://img.shields.io/badge/Flask-SaaS-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Introduction
This repository contains the complete source code, model training pipelines, and SaaS deployment architecture for an **Industrial-grade AI Inspection Platform**. Designed specifically for power grid maintenance, the system automates the detection of critical defects and anomalies using advanced Computer Vision techniques. 

The platform is engineered as a **Multi-Tenant SaaS**, featuring secure data isolation, asynchronous task queues, and dynamic hardware resource allocation to ensure production-level stability.

## ✨ Core Features

### 1. 🧠 Tri-Modular AI Detection Engine
- 🦅 **Bird Nest Detection (Mask R-CNN)**: 
  - Achieved an industrial-grade **0.907 AP50** on highly complex and cluttered backgrounds using aggressive regularization.
  - Provides pixel-level instance segmentation of bird nests on high-voltage transmission towers.
- 🛢️ **Equipment Oil Leak Detection (YOLOv8s)**: 
  - Iterated over 10 dataset versions (up to 333 images in `train10-csdn`).
  - Achieved a best F1 score of **0.518**, maintaining a strict Precision of **73.4%** while breaking the 0.40 Recall barrier, effectively conquering the challenges of small object detection in low-contrast environments.
- 🌡️ **Infrared Thermal Anomaly Detection (Hybrid OCR Pipeline)**: 
  - A deep-learning-free, pure mathematical mapping pipeline.
  - Extracts thermal legends via GPU-accelerated OCR, performs dynamic pixel-to-temperature mapping, and features an aggressive spatial/HSV **watermark suppression algorithm** to prevent camera UI interference.

### 2. ☁️ Enterprise SaaS Architecture
- **Multi-Tenant Isolation**: JWT-based authentication ensuring zero cross-tenant data leakage.
- **Asynchronous Task Queue**: Non-blocking task submissions (`PENDING` $\rightarrow$ `RUNNING` $\rightarrow$ `SUCCESS`) preventing UI freezing during heavy model inference.
- **Dynamic Thresholding**: Allows field engineers to dynamically set alarm thresholds (e.g., 50°C for thermal anomalies).
- **Analytics Dashboard**: Permanent task persistence for enterprise auditing and historical trend visualization.

---

## 📂 Repository Structure

```text
mcnn-pytorch-train/
│
├── data/                       # Datasets (Masks, YOLO labels, CSDN merged data)
├── deploy/
│   └── flask-service/          # SaaS Backend (Flask, async queue, multi-tenancy)
│       ├── app/
│       │   └── services/       # Core inference logic (model_service.py)
│       └── run.py              # Application entry point
│
├── experiments/
│   └── visualizations/         # Training curves, AP50 charts, Confusion Matrices
│
├── reports/
│   └── defense/                # Thesis defense materials (LaTeX Beamer, scripts)
│       └── Final/              # Final presentation, demo video scripts, generated plots
│
├── src/                        # Model Training Pipelines
│   ├── mask_rcnn/              # Bird Nest training scripts
│   └── oil_leak/               # YOLOv8 training scripts (runs/detect/train10-csdn)
│
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have an NVIDIA GPU with CUDA support configured.

### 2. Environment Setup
Create and activate the Conda environment:
```bash
conda create -n mask-mcnn python=3.9 -y
conda activate mask-mcnn
pip install -r deploy/flask-service/requirements.txt
```
*(Note: Ensure PyTorch is installed with the appropriate CUDA version for your system).*

---

## 🖥️ Running the SaaS Platform

### Starting the Backend Server
The SaaS backend is powered by Flask. Navigate to the deployment directory and start the service:
```bash
cd deploy/flask-service
python run.py
```
The server will start on `http://127.0.0.9:8000` (or as configured in your `run.py`).

### Using the Web UI
1. Navigate to the Login portal in your browser.
2. Authenticate as a specific tenant (e.g., Tenant A).
3. **Submit a Task**: Upload an image (Bird Nest, Oil Leak, or Infrared), configure dynamic parameters (like temperature thresholds), and click Submit.
4. **View Results**: Watch the asynchronous queue process the image and immediately view the rendered Bounding Boxes, Segmentation Masks, or Hotspot coordinates.

---

## 📊 Model Training & Reproducibility

### Oil Leak Detection (YOLOv8)
To reproduce the optimal `train10-csdn` results:
```bash
yolo task=detect mode=train model=yolov8s.pt data=data/processed/oil_leak_csdn/oil_leak_csdn.yaml epochs=150
```
To validate and generate the optimal threshold confusion matrix (Conf=0.35):
```bash
yolo val model=src/oil_leak/runs/detect/train10-csdn/weights/best.pt data=data/processed/oil_leak_csdn/oil_leak_csdn.yaml conf=0.35
```

### Infrared Watermark Suppression
The core logic for thermal mapping and UI suppression is located in:
`deploy/flask-service/app/services/model_service.py` -> `_build_border_watermark_mask()`

---

## 🎓 Academic Defense & Demo
All materials required for the final thesis defense are located in `reports/defense/Final/`.
- **`final_defense_en.tex`**: Full LaTeX Beamer presentation.
- **`speech_script.md`**: Formal 10-minute English oral presentation script.
- **`demo_video_script.md`**: Step-by-step video recording guide highlighting the engineering density of the codebase and the multi-tenant SaaS UI.
- **`generate_oil_leak_curve.py`**: Script to generate precision-recall-F1 threshold trade-off curves.

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
