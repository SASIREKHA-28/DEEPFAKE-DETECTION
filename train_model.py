"""
================================================================================
AI-Based Deepfake Detection System - Model Training Pipeline
================================================================================
This script trains a Convolutional Neural Network (CNN) using Transfer Learning
(EfficientNetB0 or Xception) to classify human face images and video frames as
either REAL (0) or FAKE (1).

Dataset Organization:
  dataset/
    ├── real/     (Authentic face images from datasets like FFHQ, CelebA, FF++)
    └── fake/     (Manipulated faces from FaceForensics++, DeepfakeDetection, etc.)

Features:
  - Automated Train / Validation splitting (stratified)
  - Pre-processing & Robust Data Augmentation (Rotation, Flips, Zoom, Shifts)
  - Fine-tuned Transfer Learning architecture with custom classification head
  - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
  - Evaluation: Generates Loss & Accuracy curves, Confusion Matrix, and
    Classification Report (Precision, Recall, F1-Score).
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Configure GPU / CPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Configured {len(gpus)} GPU(s) with memory growth.")
    except RuntimeError as e:
        print(f"[WARNING] GPU setup warning: {e}")
else:
    print("[INFO] Running on CPU.")


def build_model(input_shape=(224, 224, 3), backbone="efficientnet"):
    """
    Constructs the Deepfake Detection CNN with a pretrained backbone and custom head.

    Args:
        input_shape (tuple): Dimensions of input images (height, width, channels)
        backbone (str): 'efficientnet' for EfficientNetB0 or 'xception' for Xception

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    print(f"\n[INFO] Building Transfer Learning model using backbone: {backbone.upper()}...")

    if backbone.lower() == "xception":
        # Xception base model pretrained on ImageNet
        base_model = tf.keras.applications.Xception(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
    else:
        # EfficientNetB0 backbone (lighter, highly accurate for spatial artifacts)
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

    # Freeze base model layers initially to prevent destroying pretrained features
    base_model.trainable = False

    # Construct the classification head directly from base_model.output
    x = base_model.output
    
    # Global spatial pooling to extract salient feature vectors
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Dense reasoning layers with Batch Normalization and Dropout for regularization
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.5, name="dropout_regularization")(x)
    
    x = layers.Dense(64, activation="relu", name="dense_refinement")(x)
    x = layers.Dropout(0.3, name="dropout_refinement")(x)

    # Output node: Single sigmoid activation (0 = REAL, 1 = FAKE)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction_output")(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs, name="Deepfake_Detector_CNN")

    # Compile model with Binary Crossentropy and Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    print("[INFO] Model built successfully:")
    model.summary(print_fn=lambda x: print(f"  {x}"))
    return model, base_model


def load_datasets(dataset_dir, img_size=(224, 224), batch_size=32, validation_split=0.2, seed=42):
    """
    Loads images from dataset/real and dataset/fake using tf.keras.utils.image_dataset_from_directory.
    Applies on-the-fly data augmentation to the training split.
    """
    dataset_path = Path(dataset_dir)
    real_dir = dataset_path / "real"
    fake_dir = dataset_path / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Dataset folders missing. Ensure '{real_dir}' and '{fake_dir}' exist.\n"
            f"Place authentic images in dataset/real and manipulated images in dataset/fake."
        )

    print(f"\n[INFO] Loading dataset from: {dataset_path}")
    print(f"  - Real images dir: {real_dir}")
    print(f"  - Fake images dir: {fake_dir}")

    # Load Training Split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "fake"],  # 0: real, 1: fake
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training"
    )

    # Load Validation Split
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "fake"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
        seed=seed,
        validation_split=validation_split,
        subset="validation"
    )

    # Define Data Augmentation pipeline (essential to prevent overfitting on facial textures)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    ], name="data_augmentation")

    # Apply Augmentation and Autotuning Prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def plot_and_save_metrics(history, output_path="training_metrics.png"):
    """
    Plots training and validation Accuracy and Loss trajectories.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#3b82f6", linewidth=2.2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10b981", linewidth=2.2, linestyle="--")
    plt.title("Model Accuracy Over Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#ef4444", linewidth=2.2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#f59e0b", linewidth=2.2, linestyle="--")
    plt.title("Model Loss (Binary Crossentropy)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Accuracy & Loss curves saved to: {output_path}")


def evaluate_and_generate_reports(model, val_ds, output_dir="."):
    """
    Runs inference on the validation dataset to compute confusion matrix and classification report.
    """
    print("\n[INFO] Evaluating model on validation split...")
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true = np.array(y_true).astype(int)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Classification Report
    target_names = ["REAL (0)", "FAKE (1)"]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n================ CLASSIFICATION REPORT ================")
    print(report)
    print("=======================================================")

    report_file = Path(output_dir) / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write("Deepfake Detection Classification Report\n")
        f.write("========================================\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to: {report_file}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, fontsize=10)
    plt.yticks(tick_marks, target_names, fontsize=10)

    # Annotate counts inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {cm_path}")


def train(args):
    """
    Main training execution function.
    """
    # Create model output directory
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / "model.h5"

    # Step 1: Load Datasets
    try:
        train_ds, val_ds = load_datasets(
            dataset_dir=args.data_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            validation_split=args.val_split
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        print("\nTip: Create dataset/real and dataset/fake folders with face images before training.")
        return

    # Step 2: Build Model
    model, base_model = build_model(
        input_shape=(args.img_size, args.img_size, 3),
        backbone=args.backbone
    )

    # Step 3: Define Training Callbacks
    callbacks = [
        # Checkpoint: Save best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        # Early Stopping: Terminate training if validation loss plateaus for 5 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Dynamic Learning Rate reduction on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Step 4: Initial Training Phase (Head Training)
    print(f"\n[INFO] Starting Phase 1 Training (Feature Head) for {args.epochs} epochs...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Step 5: Optional Fine-tuning Phase (Unfreeze top layers of backbone)
    if args.fine_tune:
        print("\n[INFO] Starting Phase 2: Fine-Tuning top backbone layers...")
        base_model.trainable = True
        # Unfreeze only the top 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Low LR for fine-tuning
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
        )

        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + 5,
            initial_epoch=history_phase1.epoch[-1] + 1,
            callbacks=callbacks
        )

    # Step 6: Generate Visualization & Performance Reports
    plot_and_save_metrics(history_phase1, output_path="training_metrics.png")
    evaluate_and_generate_reports(model, val_ds, output_dir=".")

    print(f"\n[SUCCESS] Model training completed successfully! Trained model saved at: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detection CNN Model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to root dataset folder")
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=["efficientnet", "xception"], help="Model backbone")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=224, help="Input image dimension (default: 224)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set split fraction (default: 0.2)")
    parser.add_argument("--fine_tune", action="store_true", help="Enable fine-tuning of upper backbone layers")

    args = parser.parse_args()
    train(args)
