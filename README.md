# VERITAS AI — Neural Deepfake & Synthetic Media Detection System

A complete, production-ready Deepfake Detection web application featuring **Transfer Learning (EfficientNet-B0 / Xception)**, **OpenCV Facial Localization**, and **Explainable AI (Grad-CAM Heatmaps)** for images and **Temporal Frame-by-Frame Timelines** for videos.

---

## 🌟 Key Features

### 🖥️ Frontend (Vanilla HTML5, CSS3, JavaScript)
- **Modern Dark-Themed Aesthetic**: Glassmorphism, radiant gradient accents, glowing forensic badges, and responsive typography (`Outfit`, `Inter`, `JetBrains Mono`).
- **Drag-and-Drop Upload Zone**: Supports **JPG, PNG, WEBP, MP4, AVI, MOV** (up to 50 MB) with client-side format and file size validation.
- **Interactive Previews**: Instant high-resolution image preview or HTML5 video player prior to analysis.
- **Scanning Animation**: Simulated radar and neural layer propagation progress display.
- **Forensic Verdict Panel**:
  - High-visibility verdict badge: **AUTHENTIC (REAL)** in emerald green or **MANIPULATED (FAKE)** in crimson red.
  - Animated SVG circular confidence gauge with smooth easing.
  - Real vs. Fake probability distribution bar comparison.
  - **Explainable Grad-CAM Heatmap**: Side-by-side comparison of cropped face vs. Gradient-weighted Class Activation Map overlay highlighting manipulation epicenters (skin blurring, FaceSwap seams, eye warping).
  - **Temporal Video Timeline (Chart.js)**: Frame-by-frame confidence curve with 50% threshold line, total sampled frames, and flagged fake ratio.
- **Quick Demo Samples**: 1-click test buttons for authentic and synthetic samples without needing your own files.
- **"How It Works" Section**: Visual 4-step pipeline explanation.

### 🧠 Backend (FastAPI, TensorFlow, OpenCV)
- **POST `/predict`**: High-performance asynchronous endpoint accepting `multipart/form-data`.
- **Image Pipeline**: Decodes image, detects and crops face with a 15% contextual boundary margin using OpenCV Haar Cascade, normalizes to 224×224, runs CNN inference, and generates a Grad-CAM heatmap.
- **Video Pipeline**: Samples frames uniformly across the video timeline, detects faces, scores each frame with the CNN, and computes temporal consistency with average and majority voting.
- **Grad-CAM Saliency Maps**: Backpropagates class gradients to the final convolutional layer (`top_conv`), pools gradients, applies ReLU, normalizes, and blends with the face image in Base64 format.
- **Transfer Learning Model**: EfficientNet-B0 (pretrained on ImageNet) with custom GlobalAveragePooling, Dense, Dropout, and sigmoid output.
- **Turnkey Initialization & Fallback**: Automatically initializes and saves baseline weights if no custom model is found, with fallback heuristics.
- **Security & Reliability**: File size validation (< 50MB), allowed extension checks, and guaranteed temporary file cleanup using `try...finally`.

---

## 📁 Project Structure

```
deepfake-detector/
├── app.py                      # FastAPI server & inference pipelines (image, video, Grad-CAM)
├── train_model.py              # Transfer learning training pipeline with metrics & checkpointing
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & usage guide
├── test_pipeline.py            # Automated end-to-end verification script
├── model/
│   └── model.h5                # Trained / initialized CNN model weights
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphic dark theme stylesheet
│   ├── js/
│   │   └── app.js              # Drag-and-drop, gauge animation, Chart.js timeline
│   └── samples/
│       ├── sample_real.jpg     # Authentic portrait demo asset
│       ├── sample_fake.jpg     # Manipulated portrait demo asset
│       └── sample_video.mp4    # Sample video clip for temporal testing
├── templates/
│   └── index.html              # Frontend user interface
└── uploads/                    # Temporary storage for video frame decoding
```

---

## 🚀 Step-by-Step Setup & Running

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13 installed on Windows, macOS, or Linux.

### 2. Create and Activate a Virtual Environment
```bash
# Navigate to the project directory
cd C:\Users\sasir\.gemini\antigravity-ide\scratch\deepfake-detector

# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# (Or on Windows Command Prompt: venv\Scripts\activate.bat)
# (Or on Linux/macOS: source venv/bin/activate)
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Server
You can launch the server using either:
```bash
# Option A: Direct Python runner
python app.py

# Option B: Direct Uvicorn command
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Open in Your Browser
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
- Interactive Swagger API docs are available at: `http://127.0.0.1:8000/docs`
- Health check status is available at: `http://127.0.0.1:8000/health`

---

## 🎯 Training Your Own Model (`train_model.py`)

To fine-tune the model on standard deepfake benchmarks such as **FaceForensics++**, **Celeb-DF**, or Kaggle's **140k Real and Fake Faces**:

### 1. Organize Dataset
Place images into `dataset/real` and `dataset/fake`:
```
dataset/
├── real/
│   ├── face_001.jpg
│   └── face_002.jpg
└── fake/
    ├── fake_001.jpg
    └── fake_002.jpg
```

### 2. Launch Training
```bash
python train_model.py --data_dir dataset --epochs 15 --batch_size 32 --backbone efficientnet
```

### CLI Arguments:
| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `dataset` | Root path to dataset containing `real/` and `fake/` |
| `--backbone` | `efficientnet` | Backbone architecture: `efficientnet` (B0) or `xception` |
| `--epochs` | `15` | Number of training epochs |
| `--batch_size`| `32` | Batch size |
| `--img_size` | `224` | Input dimensions (224x224) |
| `--val_split` | `0.2` | Validation dataset fraction (80/20 train/val split) |
| `--fine_tune` | `False` | Enable Phase 2 fine-tuning of upper backbone layers |

### Output Artifacts:
- `model/model.h5`: Best model checkpoint saved automatically based on validation loss.
- `training_metrics.png`: Plot of Accuracy & Loss curves over epochs.
- `confusion_matrix.png`: Heatmap plot of true vs. predicted classifications.
- `classification_report.txt`: Precision, Recall, and F1-Scores.

---

## 🔬 API Reference

### `POST /predict`
Upload an image or video file for forensic inspection.

**Request:** `multipart/form-data` with `file: Binary`

**Sample Response (Image):**
```json
{
  "label": "REAL",
  "confidence": 98.4,
  "real_probability": 0.984,
  "fake_probability": 0.016,
  "heatmap_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "face_detected": true,
  "processing_time_sec": 0.28,
  "pipeline_mode": "Pretrained EfficientNet-B0 (Trained Weights)"
}
```

**Sample Response (Video):**
```json
{
  "label": "FAKE",
  "confidence": 91.7,
  "real_probability": 0.083,
  "fake_probability": 0.917,
  "frames_analyzed": 24,
  "fake_frames_count": 22,
  "fake_frame_percentage": 91.7,
  "frame_results": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "fake_probability": 0.884,
      "face_detected": true
    },
    {
      "frame_index": 15,
      "timestamp": 0.6,
      "fake_probability": 0.942,
      "face_detected": true
    }
  ],
  "processing_time_sec": 1.45,
  "pipeline_mode": "Video Sampling & Temporal Aggregation"
}
```
