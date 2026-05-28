# 🔍 Garment Hole Detection System

> Industrial-grade AI system for detecting 1–2mm holes in garments using multi-agent deep learning — developed for **Wärgon Innovation**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![CLIP](https://img.shields.io/badge/OpenCLIP-ViT--B--32-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project tackles a real industrial quality control problem: **automatically detecting 1–2mm holes in garments** before they reach customers. Tiny defects at this scale are nearly invisible to the human eye under factory conditions, making automated detection critical.

The system was designed and built end-to-end — from raw image annotation to a deployable multi-agent detection pipeline — for Wärgon Innovation, a garment manufacturing company.

---

## 🎯 Key Results

| Metric | Agent 1 (YOLOv11m) | Agent 2 (MALM-CLIP Ensemble) |
|---|---|---|
| **Precision** | ~91.7% | ~91.7% |
| **Recall** | ~83.6% | ~83.6% |
| **F1 Score** | ~0.874 | ~0.874 |
| **Detection Rate** | ~59% | ~59% |
| **Empty Precision** | ~0.920 | ~0.920 |

> ✅ Meets industrial quality control standards (Precision > 90%, Recall > 80%)

---

## 🏗️ System Architecture

```
Raw Images (380 images)
        │
        ▼
┌─────────────────────────────┐
│   Data Augmentation         │
│   • Hue shifts (±10–30°)    │
│   • Rotation (±20–30°)      │
│   • Balanced to 1,500 imgs  │
│   • COCO → YOLO conversion  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   YOLOv11m Training         │
│   • 200 epochs              │
│   • AdamW optimizer         │
│   • Cosine LR scheduling    │
│   • Train/Val/Test: 80/10/10│
└────────────┬────────────────┘
             │
        ┌────┴────┐
        ▼         ▼
  Agent 1      Agent 2
  Pure YOLO    MALM-CLIP
  (RT-DETR     (YOLOv11m +
  Ensemble)    OpenCLIP +
               Multi-agent
               Verification)
        │         │
        └────┬────┘
             ▼
    Weighted Box Fusion
    + Active Learning
    + Gradio UI
```

---

## 🧠 Technical Approach

### Notebook 1 — Data Augmentation & Training

**Problem:** Only ~380 annotated images, imbalanced (127 with holes, 253 without).

**Solution:**
- Custom augmentation pipeline with hue shifts and rotation, preserving COCO bounding box annotations
- Balanced the dataset to 750 images per class (1,500 total)
- Converted COCO → YOLO format with stratified train/val/test split
- Trained **YOLOv11m** (`yolo11m.pt`) with tuned hyperparameters

### Notebook 2 — Multi-Agent Detection System

**Agent 1 — Domain-Adaptive Ensemble:**
- YOLOv11m + RT-DETR with Test-Time Augmentation (TTA)
- Weighted Box Fusion for ensemble predictions
- Domain gap correction between augmented training data and real images

**Agent 2 — MALM-CLIP Verification:**
- YOLOv11m proposals filtered through OpenCLIP (`ViT-B-32`)
- Multi-agent verification: anomaly scoring, fabric-type-aware prompting, confidence fusion
- Active Learning module to identify hard cases and suggest re-annotation targets

**Active Learning Pipeline:**
- Uncertainty sampling across both agents
- Identifies images where agents disagree or produce low-confidence predictions
- Exports prioritised list of images for manual review and re-annotation

---

## 📁 Repository Structure

```
hole-detection-garments/
├── notebooks/
│   ├── 01_data_augmentation_training.ipynb   # Augmentation + YOLOv11m training
│   └── 02_multi_agent_detection.ipynb        # Ensemble agents + evaluation + UI
├── src/
│   ├── augmentation.py       # Core augmentation functions
│   ├── agents.py             # Agent class definitions
│   └── evaluation.py         # Evaluator classes (ProductionEvaluator, DebugEvaluator)
├── configs/
│   └── yolo_config.yaml      # YOLO training configuration
├── data/
│   └── README.md             # Dataset access instructions
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run on Google Colab (Recommended)

Both notebooks are designed to run on **Google Colab with GPU**.

1. Upload notebooks to Colab
2. Mount your Google Drive and update `base_dir` path
3. Place your dataset at the expected path (see `data/README.md`)
4. Run cells sequentially

### Quick Inference (if you have `best.pt`)

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model('your_image.jpg', conf=0.10, classes=[0])
results[0].show()
```

---

## 📦 Dataset

The dataset consists of ~1,500 images (including augmented) of garments with and without 1–2mm holes, annotated in COCO format.

**Dataset is available on request.** Please open an issue or contact via email with your use case and it will be shared directly.

> Note: Dataset was collected and annotated for Wärgon Innovation. Please use responsibly for research and educational purposes only.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Object Detection | YOLOv11m (Ultralytics) |
| Transformer Detector | RT-DETR |
| Vision-Language Model | OpenCLIP ViT-B-32 (LAION-2B) |
| Augmentation | OpenCV, Albumentations |
| Evaluation | pycocotools, custom IoU matching |
| UI | Gradio |
| Annotation Format | COCO JSON → YOLO TXT |
| Training Environment | Google Colab (T4/A100 GPU) |

---

## 📊 Evaluation Details

- **IoU Threshold:** 0.5
- **Confidence Threshold:** 0.10 (optimal), 0.15 (high precision), 0.05 (high recall)
- **Metrics:** Precision, Recall, F1, Detection Rate, Empty Image Precision (false positive rate on clean images)

The evaluation framework handles both COCO and YOLO annotation formats and includes a `DebugEvaluator` for quick sanity checks on small samples.

---

## 🔭 Future Work

- [ ] Re-train with larger annotated dataset (active learning candidates)
- [ ] Replace RT-DETR with fine-tuned version on garment domain
- [ ] Export model to ONNX/TensorRT for edge deployment
- [ ] Build REST API wrapper for factory floor integration
- [ ] Extend to other defect types (stains, tears, loose threads)

---

## 👤 Author

**Binu Shefield Shifani**

Software Engineer (5 years, Cognizant Technology Solutions)
MS AI & Automation · University West, Trollhättan, Sweden
[GitHub](https://github.com/BinuShefieldShifani)

---

## 📄 License

This project is for portfolio and research purposes. The trained model and dataset belong to Wärgon Innovation. Code is MIT licensed.
