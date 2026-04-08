"""
Synthetic Face Mesh training data generator for posture classification.

Generates realistic feature vectors that match the REAL runtime metrics
produced by PostureAnalyzer.update_from_face_mesh():

  [0] head_tilt      — degrees, sideways head tilt  (±30°)
      Calculated from ear-eye asymmetry in Face Mesh landmarks 234/454/33/263.
  [1] head_forward   — normalized forward-head score (0–0.30)
      Derived from nose deviation (landmark 1) + face-height ratio.
  [2] face_position  — Euclidean deviation of face center from frame center (0–0.50)
      face_center = mean(ear_x, nose_y, chin_y, forehead_y).
  [3] shoulder_angle — proxy shoulder-line angle from face width (radians)
      Approximated as atan2(dy, dx) of ear landmarks.
  [4] shoulder_diff  — vertical ear asymmetry (proxy for shoulder height diff)
  [5] head_height    — normalized face height / width ratio
      chin.y (152) − forehead.y (10)  divided by ear width.
  [6] ear_mean       — average Eye Aspect Ratio (auxiliary cross-modal feature)

Labels: 0 = good posture, 1 = fair, 2 = bad

Each sample is a 7-D float32 vector.  The generator creates balanced classes
with biomechanically plausible distributions so the trained PostureClassifier
works directly on the same feature space used at inference time.

Usage:
    from neurofocus.ml.posture_data_generator import generate_posture_data
    X, y = generate_posture_data(n_per_class=600)          # → (1800, 7), (1800,)
"""

import numpy as np


def generate_posture_data(
    n_per_class: int = 600,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic posture training data from Face Mesh metrics.

    Args:
        n_per_class: number of samples per class (good / fair / bad).
        seed:        random seed for reproducibility.

    Returns:
        X: numpy array, shape (n_per_class * 3, 7), dtype float32
        y: numpy array, shape (n_per_class * 3,), dtype int32
           labels: 0 = good, 1 = fair, 2 = bad
    """
    rng = np.random.default_rng(seed)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    # ──────────────────────────────────────────────────────────────
    # CLASS 0 — GOOD POSTURE
    # ──────────────────────────────────────────────────────────────
    n = n_per_class
    head_tilt      = rng.normal(0.0, 2.0, n).clip(-5, 5)          # near-zero tilt
    head_forward   = rng.normal(0.04, 0.02, n).clip(0, 0.08)     # nose well-aligned
    face_position  = rng.normal(0.06, 0.03, n).clip(0, 0.15)     # face centred
    shoulder_angle = rng.normal(0.02, 0.01, n).clip(-0.06, 0.06) # ears level
    shoulder_diff  = rng.normal(0.02, 0.01, n).clip(0, 0.05)     # symmetric
    head_height    = rng.normal(0.30, 0.04, n).clip(0.22, 0.40)  # tall posture
    ear_mean       = rng.normal(0.31, 0.02, n).clip(0.26, 0.36)  # eyes open

    X_good = np.column_stack([
        head_tilt, head_forward, face_position,
        shoulder_angle, shoulder_diff, head_height, ear_mean,
    ]).astype(np.float32)
    X_parts.append(X_good)
    y_parts.append(np.zeros(n, dtype=np.int32))

    # ──────────────────────────────────────────────────────────────
    # CLASS 1 — FAIR POSTURE  (mild degradation)
    # ──────────────────────────────────────────────────────────────
    n = n_per_class
    head_tilt      = rng.normal(0.0, 5.0, n).clip(-12, 12)
    head_forward   = rng.uniform(0.06, 0.14, n)
    face_position  = rng.uniform(0.10, 0.22, n)
    shoulder_angle = rng.uniform(-0.06, 0.06, n)
    shoulder_diff  = rng.uniform(0.03, 0.07, n)
    head_height    = rng.uniform(0.18, 0.26, n)                # slightly slouched
    ear_mean       = rng.normal(0.28, 0.03, n).clip(0.22, 0.34)

    X_fair = np.column_stack([
        head_tilt, head_forward, face_position,
        shoulder_angle, shoulder_diff, head_height, ear_mean,
    ]).astype(np.float32)
    X_parts.append(X_fair)
    y_parts.append(np.ones(n, dtype=np.int32))

    # ──────────────────────────────────────────────────────────────
    # CLASS 2 — BAD POSTURE  (two sub-types)
    # ──────────────────────────────────────────────────────────────
    n = n_per_class
    n_sideways = n // 2
    n_forward  = n - n_sideways

    # 2a — Sideways slouch  (head tilted, face shifted)
    head_tilt_s      = rng.uniform(10, 25, n_sideways) * rng.choice([-1, 1], n_sideways)
    head_forward_s   = rng.uniform(0.10, 0.22, n_sideways)
    face_position_s  = rng.uniform(0.18, 0.35, n_sideways)
    shoulder_angle_s = rng.uniform(0.06, 0.20, n_sideways) * rng.choice([-1, 1], n_sideways)
    shoulder_diff_s  = rng.uniform(0.05, 0.12, n_sideways)
    head_height_s    = rng.uniform(0.10, 0.20, n_sideways)       # compressed posture
    ear_mean_s       = rng.normal(0.25, 0.03, n_sideways).clip(0.18, 0.32)

    # 2b — Forward head  (screen lean, low face position)
    head_tilt_f      = rng.normal(0, 3, n_forward).clip(-6, 6)
    head_forward_f   = rng.uniform(0.16, 0.30, n_forward)        # nose far forward
    face_position_f  = rng.uniform(0.20, 0.40, n_forward)
    shoulder_angle_f = rng.uniform(-0.04, 0.04, n_forward)
    shoulder_diff_f  = rng.uniform(0.02, 0.06, n_forward)
    head_height_f    = rng.uniform(0.08, 0.16, n_forward)        # very compressed
    ear_mean_f       = rng.normal(0.24, 0.03, n_forward).clip(0.16, 0.30)

    X_bad = np.vstack([
        np.column_stack([head_tilt_s, head_forward_s, face_position_s,
                         shoulder_angle_s, shoulder_diff_s, head_height_s, ear_mean_s]),
        np.column_stack([head_tilt_f, head_forward_f, face_position_f,
                         shoulder_angle_f, shoulder_diff_f, head_height_f, ear_mean_f]),
    ]).astype(np.float32)
    X_parts.append(X_bad)
    y_parts.append(np.full(n, 2, dtype=np.int32))

    # ── Merge & shuffle ──────────────────────────────────────────
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def generate_fatigue_face_mesh_sequences(
    n_sequences: int = 2000,
    window_size: int = 30,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic temporal sequences matching FatigueClassifier LSTM input.

    Each sample is a (window_size, 16) array with the same features as
    TemporalFeatureExtractor._compute_features().

    Labels: 0 = awake, 1 = drowsy, 2 = sleeping

    Returns:
        X: (n_sequences * 3, window_size, 16), float32
        y: (n_sequences * 3,), int32
    """
    rng = np.random.default_rng(seed)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for label in range(3):
        for _ in range(n_sequences):
            # ── per-class dynamics ─────────────────────────────
            if label == 0:  # awake
                ear_base = rng.uniform(0.28, 0.34)
                ear_noise = 0.01
                trend = rng.uniform(-0.001, 0.001)
                blink_rate = int(rng.integers(10, 25))
                mar_base = rng.uniform(0.05, 0.15)
                mar_spikes = 0
                head_droop = rng.uniform(0.0, 0.05)
                head_tilt = rng.uniform(0.0, 0.05)
            elif label == 1:  # drowsy
                ear_base = rng.uniform(0.24, 0.30)
                ear_noise = 0.015
                trend = rng.uniform(-0.004, -0.001)
                blink_rate = int(rng.integers(25, 40))
                mar_base = rng.uniform(0.10, 0.25)
                mar_spikes = int(rng.integers(1, 4))
                head_droop = rng.uniform(0.05, 0.15)
                head_tilt = rng.uniform(0.05, 0.12)
            else:  # sleeping
                ear_base = rng.uniform(0.15, 0.22)
                ear_noise = 0.008
                trend = rng.uniform(-0.001, 0.001)
                blink_rate = int(rng.integers(0, 8))
                mar_base = rng.uniform(0.20, 0.45)
                mar_spikes = int(rng.integers(3, 8))
                head_droop = rng.uniform(0.12, 0.25)
                head_tilt = rng.uniform(0.10, 0.20)

            frames = []
            for t in range(window_size):
                ear = ear_base + trend * t + rng.normal(0, ear_noise)
                ear = np.clip(ear, 0.10, 0.40)

                mar = mar_base + rng.normal(0, 0.02)
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
                    float(ear),
                    float(ear_std),
                    float(np.clip(ear_min, 0.1, 0.4)),
                    float(np.clip(ear_max, 0.1, 0.4)),
                    float(trend),
                    float(mar),
                    float(mar),
                    float(trend * 0.5),
                    float(yawning),
                    yawn_intensity,
                    float(ear_var),
                    ear_stab,
                    float(blink_rate),
                    float(head_droop),
                    float(head_tilt),
                    float(t / window_size),
                ], dtype=np.float32)
                frames.append(feat)

            X_parts.append(np.array(frames, dtype=np.float32))
            y_parts.append(label)

    X = np.array(X_parts, dtype=np.float32)
    y = np.array(y_parts, dtype=np.int32)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


if __name__ == "__main__":
    # Quick sanity check
    X, y = generate_posture_data(n_per_class=100)
    print(f"Posture data:  X={X.shape}, y={y.shape}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Feature ranges: min={X.min(axis=0)}, max={X.max(axis=0)}")

    X2, y2 = generate_fatigue_face_mesh_sequences(n_sequences=100, window_size=30)
    print(f"\nFatigue LSTM data: X={X2.shape}, y={y2.shape}")
    print(f"  Class distribution: {np.bincount(y2)}")
