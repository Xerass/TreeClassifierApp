# Tree Species Classifier

An end-to-end computer vision project that trains a **MobileNetV3** model to classify 23 species of trees. The pipeline includes custom image preprocessing (CLAHE & Letterboxing), model optimization via **ONNX**, and a high-performance **FastAPI** backend.

## Project Overview
- **Training:** PyTorch-based pipeline using `timm` for the MobileNetV3 architecture.
- **Preprocessing:** Custom **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) for better feature extraction and **Letterbox Resizing** to preserve aspect ratios.
- **Optimization:** Models are exported to **ONNX** for faster inference and cross-platform compatibility.
- **Serving:** FastAPI server with CORS enabled for frontend integration (Next.js).

---

## Installation

### Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt