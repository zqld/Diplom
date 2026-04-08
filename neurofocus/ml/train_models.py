"""
Training pipeline for NeuroFocus models.

Provides end-to-end training and validation for:
  1. Fatigue CNN  — classifies eye region (awake / drowsy / sleeping)
  2. Fatigue LSTM — temporal sequence classifier (30-frame window, 16 features)
  3. Posture Dense — body-pose posture classifier (good / fair / bad)

All models are saved to the models/ directory after training.
Synthetic data generators are used when real labeled datasets are unavailable.

Usage:
    python -m neurofocus.ml.train_models              # train everything
    python -m neurofocus.ml.train_models --fatigue-cnn
    python -m neurofocus.ml.train_models --fatigue-lstm
    python -m neurofocus.ml.train_models --posture
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

# Project root must be on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CLASSES_FATIGUE = ["awake", "drowsy", "sleeping"]
CLASSES_POSTURE = ["good", "fair", "bad"]

# ===================================================================
# 1. FATIGUE CNN — eye-region image classifier
# ===================================================================

def _load_real_fatigue_cnn_data(data_dir: str):
    """
    Try to load real labeled eye-region images.

    Expected layout:
        data_dir/
            awake/   *.png, *.jpg …
            drowsy/  *.png, *.jpg …
            sleeping/*.png, *.jpg …

    Returns (X, y) or None if directory doesn't exist.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        return None

    X, y = [], []
    for cls_idx, cls_name in enumerate(CLASSES_FATIGUE):
        cls_dir = dir_path / cls_name
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            X.append(img.astype(np.float32) / 255.0)
            y.append(cls_idx)

    if len(X) == 0:
        return None
    return np.array(X), np.array(y, dtype=np.int32)


def generate_fatigue_cnn_synthetic(samples_per_class: int = 500, seed: int = 42):
    """
    Generate synthetic 64x64 grayscale eye-region images.

    Heuristic:
      - awake:    bright horizontal slit (open eye), high variance
      - drowsy:   narrower slit, partial occlusion by upper eyelid
      - sleeping: almost entirely dark (closed eyelid)

    Heavy augmentation to prevent overfitting: random shifts, rotation,
    brightness, contrast, blur, noise, skin-tone overlay.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for cls_idx in range(3):
        for _ in range(samples_per_class):
            img = np.zeros((64, 64), dtype=np.float32)

            # ── Random geometry ──────────────────────────────
            cx = 32 + int(rng.integers(-8, 8))
            cy = 32 + int(rng.integers(-8, 8))
            scale = rng.uniform(0.7, 1.3)

            if cls_idx == 0:  # awake — wide open iris
                iris_r = int(rng.integers(8, 18) * scale)
                # Iris as ellipse (more realistic than circle)
                rx = iris_r
                ry = int(iris_r * rng.uniform(0.6, 1.0))
                cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, 0.8, -1)
                # Pupil
                pupil_r = max(1, int(iris_r * 0.4))
                cv2.circle(img, (cx, cy), pupil_r, 0.2, -1)
                # Eyelid lines (thick, noisy)
                lid_thick = rng.integers(1, 4)
                cv2.line(img, (cx - rx - 4, cy - ry - 2),
                         (cx + rx + 4, cy - ry - 2), 1.0, lid_thick)
                cv2.line(img, (cx - rx - 4, cy + ry + 2),
                         (cx + rx + 4, cy + ry + 2), 1.0, lid_thick)
                # Skin texture
                skin_val = rng.uniform(0.1, 0.3)
                mask = (img == 0)
                if mask.any():
                    img[mask] = skin_val + rng.normal(0, 0.05, mask.sum())

            elif cls_idx == 1:  # drowsy — half-closed
                iris_r = int(rng.integers(5, 12) * scale)
                cv2.ellipse(img, (cx, cy), (iris_r, int(iris_r * 0.5)), 0, 0, 360, 0.6, -1)
                # Upper eyelid lowered significantly
                lid_y = cy - int(iris_r * 0.3) + int(rng.integers(-3, 3))
                lid_thick = rng.integers(2, 5)
                cv2.rectangle(img, (0, 0), (64, lid_y), 0.15, -1)
                cv2.line(img, (2, lid_y), (62, lid_y), 0.5, lid_thick)
                # Lower lid visible
                lower_y = cy + iris_r + rng.integers(-2, 2)
                if lower_y < 64:
                    cv2.line(img, (2, lower_y), (62, lower_y), 0.3, rng.integers(1, 3))

            else:  # sleeping — dark uniform with faint eyelid crease
                base = rng.uniform(0.05, 0.25)
                img[:] = base
                # Eyelid crease line (slightly curved)
                lid_y = cy + int(rng.integers(-5, 5))
                pts = np.array([
                    [2, lid_y + int(rng.integers(-3, 3))],
                    [16, lid_y + int(rng.integers(-2, 2))],
                    [32, lid_y + int(rng.integers(-1, 1))],
                    [48, lid_y + int(rng.integers(-2, 2))],
                    [62, lid_y + int(rng.integers(-3, 3))],
                ], dtype=np.int32)
                cv2.polylines(img, [pts], False, 0.4, rng.integers(1, 3))
                # Eyelash hints
                for lx in range(8, 56, rng.integers(4, 8)):
                    ly = lid_y + rng.integers(-2, 2)
                    cv2.line(img, (lx, ly), (lx, ly - rng.integers(2, 5)), 0.2, 1)

            # ── Augmentation ────────────────────────────────
            # Gaussian blur (simulate out-of-focus)
            if rng.random() < 0.4:
                k = rng.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)

            # Brightness & contrast
            img = img * rng.uniform(0.6, 1.4) + rng.uniform(-0.1, 0.1)

            # Random noise
            img += rng.normal(0, rng.uniform(0.02, 0.08), img.shape).astype(np.float32)

            # Vignette (dark corners)
            if rng.random() < 0.3:
                ys, xs = np.ogrid[:64, :64]
                cx2, cy2 = 32, 32
                dist = ((xs - cx2) ** 2 + (ys - cy2) ** 2) / (32 ** 2)
                img *= (1.0 - 0.3 * dist)

            img = np.clip(img, 0.0, 1.0)
            X.append(img)
            y.append(cls_idx)

    X = np.array(X).reshape(-1, 64, 64, 1)
    y = np.array(y, dtype=np.int32)
    return X, y


def build_fatigue_cnn(input_shape=(64, 64, 1)):
    """Build a SMALL CNN to prevent overfitting on synthetic data."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(CLASSES_FATIGUE), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_fatigue_cnn(
    data_dir: str | None = None,
    samples_per_class: int = 500,
    epochs: int = 40,
    batch_size: int = 64,
    save_path: str | None = None,
):
    """Train fatigue CNN. Falls back to synthetic data if no real data found."""
    print("=" * 60)
    print("  FATIGUE CNN — Training")
    print("=" * 60)

    # Load data
    if data_dir:
        result = _load_real_fatigue_cnn_data(data_dir)
    else:
        result = None

    if result is not None:
        X, y = result
        print(f"[+] Loaded {len(X)} real images from {data_dir}")
    else:
        print("[!] Real data not found — generating synthetic dataset")
        X, y = generate_fatigue_cnn_synthetic(samples_per_class)
        print(f"[+] Generated {len(X)} synthetic images "
              f"({samples_per_class} per class)")

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Train/val split 80/20
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"    Train: {len(X_train)} | Val: {len(X_val)}")

    model = build_fatigue_cnn()

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Data augmentation — on-the-fly transforms during training
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        shear_range=0.1,
        horizontal_flip=False,
        brightness_range=[0.5, 1.5],
        fill_mode="nearest",
    )

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor="val_loss"),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[+] Validation accuracy: {val_acc:.4f}")

    # Save
    out = save_path or str(MODELS_DIR / "fatigue_cnn.keras")
    model.save(out)
    print(f"[+] Model saved → {out}")

    # Save training report
    report = {
        "model": "fatigue_cnn",
        "samples": len(X),
        "synthetic": result is None,
        "epochs": epochs,
        "batch_size": batch_size,
        "val_accuracy": round(val_acc, 4),
        "val_loss": round(val_loss, 4),
        "saved_to": out,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_report(report, "fatigue_cnn_report.json")
    return model, history


# ===================================================================
# 2. FATIGUE LSTM — temporal sequence classifier
# ===================================================================

def generate_fatigue_lstm_sequences(
    n_sequences: int = 2000,
    window_size: int = 30,
    seed: int = 42,
):
    """
    Generate synthetic temporal sequences for fatigue LSTM.

    Each sequence = (window_size, 16) feature vector.
    Labels: 0=awake, 1=drowsy, 2=sleeping

    Features match TemporalFeatureExtractor output:
      [0]  ear_mean, [1] ear_std, [2] ear_min, [3] ear_max,
      [4]  ear_trend, [5] mar_mean, [6] mar_max, [7] mar_trend,
      [8]  yawning, [9] yawn_intensity, [10] ear_variance_long,
      [11] ear_stability, [12] estimated_blink_rate,
      [13] head_droop_mean, [14] head_tilt_mean, [15] time_elapsed
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []

    for label in range(3):
        for _ in range(n_sequences):
            # Base EAR dynamics per class
            if label == 0:  # awake — stable high EAR
                ear_base = rng.uniform(0.28, 0.34)
                ear_noise = 0.01
                trend = rng.uniform(-0.001, 0.001)
                blink_rate = rng.integers(10, 25)
                mar_base = rng.uniform(0.05, 0.15)
                mar_spikes = 0
                head_droop = rng.uniform(0.0, 0.05)
                head_tilt = rng.uniform(0.0, 0.05)
            elif label == 1:  # drowsy — declining EAR, more blinks
                ear_base = rng.uniform(0.24, 0.30)
                ear_noise = 0.015
                trend = rng.uniform(-0.004, -0.001)
                blink_rate = rng.integers(25, 40)
                mar_base = rng.uniform(0.10, 0.25)
                mar_spikes = rng.integers(1, 4)
                head_droop = rng.uniform(0.05, 0.15)
                head_tilt = rng.uniform(0.05, 0.12)
            else:  # sleeping — very low EAR, high MAR, no trend
                ear_base = rng.uniform(0.15, 0.22)
                ear_noise = 0.008
                trend = rng.uniform(-0.001, 0.001)
                blink_rate = rng.integers(0, 8)
                mar_base = rng.uniform(0.20, 0.45)
                mar_spikes = rng.integers(3, 8)
                head_droop = rng.uniform(0.12, 0.25)
                head_tilt = rng.uniform(0.10, 0.20)

            frames = []
            for t in range(window_size):
                ear = ear_base + trend * t + rng.normal(0, ear_noise)
                ear = np.clip(ear, 0.10, 0.40)

                mar = mar_base + rng.normal(0, 0.02)
                # Insert occasional MAR spikes (yawns)
                if mar_spikes > 0 and rng.random() < 0.1:
                    mar += rng.uniform(0.2, 0.4)
                    mar_spikes -= 1
                mar = np.clip(mar, 0.0, 1.0)

                ear_std = ear_noise * (1 + 0.5 * rng.random())
                ear_min = ear - rng.uniform(0.01, 0.03)
                ear_max = ear + rng.uniform(0.01, 0.03)
                ear_var = ear_std ** 2
                ear_stab = float(np.clip(1.0 - ear_std * 10, 0, 1))

                yawning = bool(mar > 0.5)
                yawn_intensity = float(max(0, mar - 0.3))

                feat = np.array([
                    float(ear),               # 0  ear_mean
                    float(ear_std),            # 1  ear_std
                    float(np.clip(ear_min, 0.1, 0.4)),  # 2  ear_min
                    float(np.clip(ear_max, 0.1, 0.4)),  # 3  ear_max
                    float(trend),              # 4  ear_trend
                    float(mar),                # 5  mar_mean
                    float(mar),                # 6  mar_max  (proxy)
                    float(trend * 0.5),        # 7  mar_trend
                    float(yawning),            # 8  yawning
                    yawn_intensity,             # 9  yawn_intensity
                    float(ear_var),            # 10 ear_variance_long
                    ear_stab,                   # 11 ear_stability
                    float(blink_rate),          # 12 estimated_blink_rate
                    float(head_droop),          # 13 head_droop_mean
                    float(head_tilt),           # 14 head_tilt_mean
                    float(t / window_size),     # 15 time_elapsed
                ], dtype=np.float32)
                frames.append(feat)

            X_parts.append(np.array(frames, dtype=np.float32))
            y_parts.append(label)

    X = np.array(X_parts, dtype=np.float32)
    y = np.array(y_parts, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def build_fatigue_lstm(input_shape=(30, 16)):
    """Build LSTM model for temporal fatigue classification."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES_FATIGUE), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_fatigue_lstm(
    n_sequences: int = 2000,
    window_size: int = 30,
    epochs: int = 30,
    batch_size: int = 64,
    save_path: str | None = None,
):
    """Train fatigue LSTM on synthetic temporal sequences."""
    print("=" * 60)
    print("  FATIGUE LSTM — Training")
    print("=" * 60)

    print("[+] Generating synthetic temporal sequences …")
    X, y = generate_fatigue_lstm_sequences(n_sequences, window_size)
    print(f"[+] Generated {len(X)} sequences ({n_sequences} per class)")

    # Shuffle & split 80/20
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"    Train: {len(X_train)} | Val: {len(X_val)}")

    model = build_fatigue_lstm(input_shape=(window_size, 16))

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, monitor="val_loss"),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[+] Validation accuracy: {val_acc:.4f}")

    out = save_path or str(MODELS_DIR / "fatigue_lstm.keras")
    model.save(out)
    print(f"[+] Model saved → {out}")

    report = {
        "model": "fatigue_lstm",
        "sequences": len(X),
        "window_size": window_size,
        "features": 16,
        "synthetic": True,
        "epochs": epochs,
        "batch_size": batch_size,
        "val_accuracy": round(val_acc, 4),
        "val_loss": round(val_loss, 4),
        "saved_to": out,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_report(report, "fatigue_lstm_report.json")
    return model, history


# ===================================================================
# 3. POSTURE DENSE — body-pose / Face Mesh posture classifier
# ===================================================================

def generate_posture_face_mesh_features(
    n_per_class: int = 600,
    seed: int = 42,
):
    """
    Generate synthetic posture features matching the REAL runtime pipeline.

    Features (7 dims) align with PostureClassifier._predict_geometric():
      [0] head_tilt      — degrees, tilt of head sideways
      [1] head_forward   — normalized forward head protrusion
      [2] face_position  — deviation from frame center
      [3] shoulder_angle — shoulder line angle (when pose visible)
      [4] shoulder_diff  — vertical shoulder asymmetry
      [5] head_height    — nose-to-shoulders vertical distance
      [6] ear_mean       — auxiliary EAR for cross-modal fusion

    Labels: 0=good, 1=fair, 2=bad
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []

    # --- GOOD POSTURE ---
    n = n_per_class
    head_tilt     = rng.normal(0.0, 2.0, n).clip(-5, 5)          # degrees
    head_forward  = rng.normal(0.04, 0.02, n).clip(0, 0.08)
    face_position = rng.normal(0.06, 0.03, n).clip(0, 0.15)
    shoulder_angle = rng.normal(1.5, 1.5, n).clip(0, 6)
    shoulder_diff = rng.normal(0.02, 0.01, n).clip(0, 0.05)
    head_height   = rng.normal(0.30, 0.04, n).clip(0.22, 0.40)
    ear_mean      = rng.normal(0.31, 0.02, n).clip(0.26, 0.36)

    X_good = np.column_stack([
        head_tilt, head_forward, face_position,
        shoulder_angle, shoulder_diff, head_height, ear_mean,
    ]).astype(np.float32)
    X_parts.append(X_good)
    y_parts.append(np.zeros(n, dtype=np.int32))

    # --- FAIR POSTURE ---
    n = n_per_class
    head_tilt     = rng.normal(0.0, 5.0, n).clip(-12, 12)
    head_forward  = rng.uniform(0.06, 0.14, n)
    face_position = rng.uniform(0.10, 0.22, n)
    shoulder_angle = rng.uniform(5, 12, n)
    shoulder_diff = rng.uniform(0.03, 0.07, n)
    head_height   = rng.uniform(0.18, 0.26, n)
    ear_mean      = rng.normal(0.28, 0.03, n).clip(0.22, 0.34)

    X_fair = np.column_stack([
        head_tilt, head_forward, face_position,
        shoulder_angle, shoulder_diff, head_height, ear_mean,
    ]).astype(np.float32)
    X_parts.append(X_fair)
    y_parts.append(np.ones(n, dtype=np.int32))

    # --- BAD POSTURE ---
    n = n_per_class
    n_sideways = n // 2
    n_forward  = n - n_sideways

    # Sideways slouch
    head_tilt_s     = rng.uniform(10, 25, n_sideways) * rng.choice([-1, 1], n_sideways)
    head_forward_s  = rng.uniform(0.10, 0.22, n_sideways)
    face_position_s = rng.uniform(0.18, 0.35, n_sideways)
    shoulder_angle_s = rng.uniform(10, 22, n_sideways)
    shoulder_diff_s = rng.uniform(0.05, 0.12, n_sideways)
    head_height_s   = rng.uniform(0.10, 0.20, n_sideways)
    ear_mean_s      = rng.normal(0.25, 0.03, n_sideways).clip(0.18, 0.32)

    # Forward head
    head_tilt_f     = rng.normal(0, 3, n_forward).clip(-6, 6)
    head_forward_f  = rng.uniform(0.16, 0.30, n_forward)
    face_position_f = rng.uniform(0.20, 0.40, n_forward)
    shoulder_angle_f = rng.uniform(4, 10, n_forward)
    shoulder_diff_f = rng.uniform(0.02, 0.06, n_forward)
    head_height_f   = rng.uniform(0.08, 0.16, n_forward)
    ear_mean_f      = rng.normal(0.24, 0.03, n_forward).clip(0.16, 0.30)

    X_bad = np.vstack([
        np.column_stack([head_tilt_s, head_forward_s, face_position_s,
                         shoulder_angle_s, shoulder_diff_s, head_height_s, ear_mean_s]),
        np.column_stack([head_tilt_f, head_forward_f, face_position_f,
                         shoulder_angle_f, shoulder_diff_f, head_height_f, ear_mean_f]),
    ]).astype(np.float32)
    X_parts.append(X_bad)
    y_parts.append(np.full(n, 2, dtype=np.int32))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def build_posture_dense(input_shape=(7,)):
    """Build Dense model for posture classification."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(16, activation="relu"),
        layers.Dense(len(CLASSES_POSTURE), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_posture(
    n_per_class: int = 600,
    epochs: int = 40,
    batch_size: int = 64,
    save_path: str | None = None,
):
    """Train posture classifier on synthetic Face Mesh features."""
    print("=" * 60)
    print("  POSTURE DENSE — Training")
    print("=" * 60)

    print("[+] Generating synthetic Face Mesh posture features …")
    X, y = generate_posture_face_mesh_features(n_per_class)
    print(f"[+] Generated {len(X)} feature vectors ({n_per_class} per class)")

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"    Train: {len(X_train)} | Val: {len(X_val)}")

    model = build_posture_dense(input_shape=(7,))

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor="val_loss"),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[+] Validation accuracy: {val_acc:.4f}")

    out = save_path or str(MODELS_DIR / "posture_model.keras")
    model.save(out)
    print(f"[+] Model saved → {out}")

    # Also save as posture_lstm.keras for compatibility
    lstm_path = str(MODELS_DIR / "posture_lstm.keras")
    model.save(lstm_path)
    print(f"[+] Also saved → {lstm_path}")

    report = {
        "model": "posture_dense",
        "features": 7,
        "feature_names": [
            "head_tilt", "head_forward", "face_position",
            "shoulder_angle", "shoulder_diff", "head_height", "ear_mean",
        ],
        "samples": len(X),
        "synthetic": True,
        "epochs": epochs,
        "batch_size": batch_size,
        "val_accuracy": round(val_acc, 4),
        "val_loss": round(val_loss, 4),
        "saved_to": out,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_report(report, "posture_report.json")
    return model, history


# ===================================================================
# Helpers
# ===================================================================

def _save_report(report: dict, filename: str):
    """Save training report to models/ directory."""
    path = MODELS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[+] Report saved → {path}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Train NeuroFocus models")
    parser.add_argument("--fatigue-cnn", action="store_true",
                        help="Train fatigue CNN")
    parser.add_argument("--fatigue-lstm", action="store_true",
                        help="Train fatigue LSTM")
    parser.add_argument("--posture", action="store_true",
                        help="Train posture Dense model")
    parser.add_argument("--all", action="store_true",
                        help="Train all models")
    parser.add_argument("--epochs-cnn", type=int, default=40)
    parser.add_argument("--epochs-lstm", type=int, default=30)
    parser.add_argument("--epochs-posture", type=int, default=40)
    parser.add_argument("--fatigue-data-dir", type=str, default=None,
                        help="Path to real eye-region image dataset")
    args = parser.parse_args()

    if not any([args.fatigue_cnn, args.fatigue_lstm, args.posture, args.all]):
        args.all = True  # default: train everything

    if args.fatigue_cnn or args.all:
        train_fatigue_cnn(
            data_dir=args.fatigue_data_dir,
            epochs=args.epochs_cnn,
        )

    if args.fatigue_lstm or args.all:
        train_fatigue_lstm(epochs=args.epochs_lstm)

    if args.posture or args.all:
        train_posture(epochs=args.epochs_posture)

    print("\n" + "=" * 60)
    print("  All training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
