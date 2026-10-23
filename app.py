"""
================================================================================
VERITAS AI - Deepfake & Synthetic Media Detection System
================================================================================
Backend Server powered by FastAPI, TensorFlow / Keras (EfficientNet-B0),
OpenCV Face Localization, and Explainable Grad-CAM Heatmap Generation.

Key Endpoints:
  - GET  /         : Serves the web application user interface
  - POST /predict  : Multipart file upload endpoint (Image or Video)
  - GET  /health   : Server and model status endpoint
================================================================================
"""

import os
import sys
import time
import base64
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("VeritasAI")

# ------------------------------------------------------------------------------
# Application & Constants Initialization
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_DIR = BASE_DIR / "model"
UPLOADS_DIR = BASE_DIR / "uploads"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.h5"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov", ".mkv"}

app = FastAPI(
    title="Veritas AI - Deepfake Detection System",
    description="Advanced Neural Deepfake Detection with Grad-CAM and Temporal Video Timeline",
    version="2.0.0"
)

# Enable CORS for cross-origin interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global Model & Detector References
model: Optional[keras.Model] = None
is_demo_mode: bool = False
face_cascade = None


# ------------------------------------------------------------------------------
# Model Architecture & Initialization
# ------------------------------------------------------------------------------
def build_transfer_cnn_model(input_shape=(224, 224, 3)) -> keras.Model:
    """
    Constructs the EfficientNetB0 Transfer Learning CNN architecture
    with custom classification head for Deepfake detection.
    """
    logger.info("Instantiating EfficientNet-B0 backbone with ImageNet pretrained weights...")
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout_regularization")(x)
    x = layers.Dense(64, activation="relu", name="dense_refinement")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction_output")(x)

    cnn_model = keras.Model(inputs=base_model.input, outputs=outputs, name="Deepfake_Classifier")
    return cnn_model


def initialize_face_detector():
    """
    Initializes OpenCV Haar Cascade frontal face classifier safely.
    """
    global face_cascade
    try:
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if not face_cascade.empty():
                    logger.info("OpenCV Haar Cascade Face Detector loaded successfully.")
                    return
        logger.warning("Haar Cascade not directly available; will use adaptive center face ROI.")
    except Exception as e:
        logger.warning(f"Face detector initialization warning: {e}. Fallback enabled.")
        face_cascade = None


@app.on_event("startup")
def startup_event():
    """
    Loads trained model from model/model.h5 if present;
    otherwise initializes the Transfer Learning CNN in Demo/Fallback mode.
    """
    global model, is_demo_mode
    initialize_face_detector()

    if MODEL_PATH.exists():
        try:
            logger.info(f"Loading custom trained model from: {MODEL_PATH}")
            model = keras.models.load_model(str(MODEL_PATH))
            is_demo_mode = False
            logger.info("Custom Deepfake Detection model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading {MODEL_PATH}: {e}. Switching to DEMO fallback mode.")
            model = build_transfer_cnn_model()
            is_demo_mode = True
    else:
        logger.warning("=" * 70)
        logger.warning("[WARNING] No trained model file found at 'model/model.h5'!")
        logger.warning("[INFO] Initializing EfficientNet-B0 CNN architecture in DEMO / FALLBACK mode.")
        logger.warning("[INFO] Run 'python train_model.py' to train on your custom dataset.")
        logger.warning("=" * 70)
        model = build_transfer_cnn_model()
        is_demo_mode = True

        # Save initial baseline weights so model/model.h5 exists
        try:
            model.save(str(MODEL_PATH))
            logger.info(f"Initialized and saved baseline architecture at: {MODEL_PATH}")
        except Exception as err:
            logger.warning(f"Could not write default baseline model.h5: {err}")


# ------------------------------------------------------------------------------
# Image Preprocessing & Face Localization
# ------------------------------------------------------------------------------
def detect_and_crop_face(image_bgr: np.ndarray, target_size=(224, 224)) -> tuple[np.ndarray, bool]:
    """
    Detects the human face in an image using OpenCV, applies a 15% margin
    around the bounding box to capture blending boundaries, and crops to 224x224.
    If no face is detected, returns a center crop with face_detected=False.
    """
    h, w = image_bgr.shape[:2]
    face_detected = False
    cropped_face = None

    if face_cascade is not None and not face_cascade.empty():
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) > 0:
                # Select the largest face by area
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, fw, fh = largest_face

                # Add a 15% contextual margin to capture facial seams / hairline
                x_margin = int(fw * 0.15)
                y_margin = int(fh * 0.15)

                x1 = max(0, x - x_margin)
                y1 = max(0, y - y_margin)
                x2 = min(w, x + fw + x_margin)
                y2 = min(h, y + fh + y_margin)

                cropped_face = image_bgr[y1:y2, x1:x2]
                face_detected = True
        except Exception as e:
            logger.warning(f"Haar face detection error: {e}")

    # Fallback to center square crop if face wasn't located
    if cropped_face is None or cropped_face.size == 0:
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped_face = image_bgr[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Resize to model input dimensions (224, 224)
    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
    return resized_face, face_detected


# ------------------------------------------------------------------------------
# Explainable AI: Grad-CAM Saliency Heatmap Generation
# ------------------------------------------------------------------------------
def generate_gradcam_heatmap(img_tensor: np.ndarray, cnn_model: keras.Model, last_conv_name="top_conv") -> np.ndarray:
    """
    Generates a Gradient-weighted Class Activation Map (Grad-CAM)
    highlighting the discriminative spatial regions that led the CNN
    to suspect manipulation.
    """
    try:
        # Locate target convolutional layer (top_conv or deepest conv layer)
        conv_layer = None
        try:
            conv_layer = cnn_model.get_layer(last_conv_name)
        except Exception:
            conv_candidates = [l for l in cnn_model.layers if "conv" in l.name.lower()]
            if conv_candidates:
                conv_layer = conv_candidates[-1]

        if conv_layer is not None:
            grad_model = keras.models.Model(
                inputs=cnn_model.input,
                outputs=[conv_layer.output, cnn_model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                class_channel = predictions[:, 0]

            # Compute gradients of the prediction w.r.t feature maps
            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Apply ReLU to keep only positive contributions
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val

            return heatmap.numpy()
    except Exception as e:
        logger.warning(f"Grad-CAM gradient computation notice: {e}. Generating spatial feature saliency.")

    # Fallback to feature map variance if gradients are flat or layer unavailable
    try:
        conv_candidates = [l for l in cnn_model.layers if "conv" in l.name.lower()]
        target_l = conv_candidates[-1] if conv_candidates else cnn_model.layers[-3]
        feat_model = keras.models.Model(inputs=cnn_model.input, outputs=target_l.output)
        features = feat_model(img_tensor)[0]
        heatmap = tf.reduce_mean(features, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()
    except Exception as fallback_err:
        logger.warning(f"Feature saliency fallback error: {fallback_err}. Using gaussian kernel center.")
        # Final graceful fallback: radial kernel
        y, x = np.ogrid[-1:1:7j, -1:1:7j]
        h = np.exp(-(x**2 + y**2) / 0.5)
        return h.astype(np.float32)


def overlay_heatmap(face_bgr: np.ndarray, heatmap: np.ndarray, alpha=0.45) -> str:
    """
    Overlays the Grad-CAM heatmap onto the face BGR image and encodes to Base64 JPEG.
    """
    h, w = face_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

    # Apply JET colormap (Blue=Natural, Green/Yellow=Anomalous, Red=Severe Artifacts)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap on authentic cropped face
    superimposed = cv2.addWeighted(face_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)

    # Convert to Base64 string
    _, buffer = cv2.imencode(".jpg", superimposed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


# ------------------------------------------------------------------------------
# Image Inference Pipeline
# ------------------------------------------------------------------------------
def evaluate_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Full inference pipeline for single image input:
      1. Decode image with OpenCV
      2. Detect and crop facial region
      3. Feed normalized tensor to CNN
      4. Compute Grad-CAM heatmap overlay
      5. Formulate verdict and probability scores
    """
    start_time = time.time()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Invalid image file or unsupported image encoding.")

    # Step 1: Detect and crop face
    face_bgr, face_detected = detect_and_crop_face(image_bgr)

    # Step 2: Convert to RGB and prepare model input tensor
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = np.expand_dims(face_rgb.astype(np.float32), axis=0)

    # Step 3: Run CNN inference
    pred_raw = float(model.predict(img_tensor, verbose=0)[0][0])

    # If in DEMO / FALLBACK mode (before user training on FF++),
    # complement with subtle spatial-frequency artifact heuristics (Laplacian edge blur & noise disparity)
    if is_demo_mode:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Generative deepfakes often suffer from frequency smoothing or unnatural high-frequency noise
        noise_diff = np.std(image_bgr[:, :, 0] - image_bgr[:, :, 2])
        simulated_bias = 0.5 + 0.3 * np.tanh((laplacian_var - 250.0) / 300.0)
        # Blend CNN prediction with frequency heuristic for demonstration
        fake_prob = float(np.clip(0.55 * pred_raw + 0.45 * (1.0 - simulated_bias), 0.02, 0.98))
    else:
        fake_prob = float(np.clip(pred_raw, 0.0, 1.0))

    real_prob = float(1.0 - fake_prob)
    label = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = float(round((fake_prob if label == "FAKE" else real_prob) * 100.0, 2))

    # Step 4: Generate Grad-CAM Heatmap
    heatmap = generate_gradcam_heatmap(img_tensor, model)
    heatmap_b64 = overlay_heatmap(face_bgr, heatmap)

    elapsed_sec = round(time.time() - start_time, 3)

    return {
        "label": label,
        "confidence": confidence,
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4),
        "heatmap_image": heatmap_b64,
        "face_detected": face_detected,
        "processing_time_sec": elapsed_sec,
        "pipeline_mode": "Pretrained EfficientNet-B0 (Trained Weights)" if not is_demo_mode else "Demo / Fallback Mode (Untrained)"
    }


# ------------------------------------------------------------------------------
# Video Inference Pipeline
# ------------------------------------------------------------------------------
def evaluate_video(video_path: str, max_frames: int = 24) -> Dict[str, Any]:
    """
    Video analysis pipeline:
      1. Extract sampled frames uniformly across video duration
      2. Run face localization and CNN prediction on each frame
      3. Track frame-by-frame temporal manipulation scores
      4. Aggregate with average and majority vote for final verdict
    """
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file for frame extraction.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames <= 0:
        total_frames = 100  # Fallback estimate

    # Select sampled frame indices uniformly across the video
    num_samples = min(max_frames, max(4, total_frames // 4))
    sample_indices = np.linspace(0, max(0, total_frames - 1), num=num_samples, dtype=int)

    frame_results = []
    fake_probs_list = []
    fake_count = 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Detect face & resize
        face_bgr, has_face = detect_and_crop_face(frame)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = np.expand_dims(face_rgb.astype(np.float32), axis=0)

        # CNN inference
        pred_raw = float(model.predict(img_tensor, verbose=0)[0][0])
        fake_prob = float(np.clip(pred_raw, 0.02, 0.98))

        if fake_prob > 0.5:
            fake_count += 1

        fake_probs_list.append(fake_prob)
        timestamp_sec = round(float(idx) / fps, 2)

        frame_results.append({
            "frame_index": int(idx),
            "timestamp": timestamp_sec,
            "fake_probability": round(fake_prob, 4),
            "face_detected": has_face
        })

    cap.release()

    if not fake_probs_list:
        raise ValueError("No valid frames could be decoded from the video.")

    # Aggregate Predictions (Mean & Majority Voting)
    avg_fake_prob = float(np.mean(fake_probs_list))
    avg_real_prob = float(1.0 - avg_fake_prob)
    frames_analyzed = len(fake_probs_list)
    fake_ratio = round((fake_count / frames_analyzed) * 100.0, 1)

    # Final verdict: Flagged as FAKE if mean probability >= 0.5 or >= 50% frames flagged
    is_fake = (avg_fake_prob >= 0.5) or (fake_ratio >= 50.0)
    label = "FAKE" if is_fake else "REAL"
    winning_prob = avg_fake_prob if is_fake else avg_real_prob
    winning_ratio = (fake_ratio if is_fake else (100.0 - fake_ratio)) / 100.0
    confidence = float(round(max(winning_prob, winning_ratio) * 100.0, 2))
    elapsed_sec = round(time.time() - start_time, 3)

    return {
        "label": label,
        "confidence": confidence,
        "real_probability": round(avg_real_prob, 4),
        "fake_probability": round(avg_fake_prob, 4),
        "frames_analyzed": frames_analyzed,
        "fake_frames_count": fake_count,
        "fake_frame_percentage": fake_ratio,
        "frame_results": frame_results,
        "processing_time_sec": elapsed_sec,
        "pipeline_mode": "Video Sampling & Temporal Aggregation"
    }


from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
async def serve_index():
    """
    Serves the main application landing page.
    """
    index_file = TEMPLATES_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/health")
async def health_check():
    """
    Returns server status, model mode, and platform details.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "demo_mode": is_demo_mode,
        "backend": "FastAPI + TensorFlow 2.21 + OpenCV"
    }


@app.post("/predict")
async def predict_media(file: UploadFile = File(...)):
    """
    Main prediction endpoint:
      - Validates file type and size (< 50MB)
      - Routes images to evaluate_image (with Grad-CAM)
      - Routes videos to evaluate_video (with Temporal Timeline)
      - Guarantees temporary file cleanup
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read uploaded bytes
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    # Validate file size
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )

    is_video = ext in {".mp4", ".avi", ".mov", ".mkv"}

    # Image Processing Branch
    if not is_video:
        try:
            result = evaluate_image(contents)
            return JSONResponse(content=result)
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

    # Video Processing Branch (Safely handles temporary file)
    temp_video = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf_file:
            tf_file.write(contents)
            temp_video_path = tf_file.name

        result = evaluate_video(temp_video_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        # Guarantee cleanup of temporary video files
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete temp video file {temp_video_path}: {cleanup_err}")


# ------------------------------------------------------------------------------
# Direct Execution Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Veritas AI Deepfake Detection Server on http://127.0.0.1:8000...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
