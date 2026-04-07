"""
Synthetic posture training data generator.

Generates realistic feature vectors for posture classification
based on biomechanical constraints. Used to train PostureClassifier
when real labeled data is not available.

Features match extract_pose_features() in preprocessing.py:
  [0] shoulder_angle      - radians, ~[-0.5, 0.5]
  [1] shoulder_y_diff     - pixels, ~[0, 60]
  [2] shoulder_width/100  - normalized shoulder width
  [3] forward_lean/50     - head position relative to shoulders
  [4] hip_y_diff          - pixels, ~[0, 40]
  [5] torso_tilt/50       - lateral lean
  [6] torso_length/100    - torso height

Labels: 0=good, 1=fair, 2=bad
"""

import numpy as np


def generate_posture_data(n_per_class: int = 600, seed: int = 42) -> tuple:
    """
    Generate synthetic posture training data.

    Returns:
        X: numpy array, shape (n_per_class*3, 7)
        y: numpy array, shape (n_per_class*3,), dtype int
    """
    rng = np.random.default_rng(seed)
    X_parts = []
    y_parts = []

    # ----- CLASS 0: GOOD POSTURE -----
    n = n_per_class
    shoulder_angle  = rng.normal(0.0, 0.02, n).clip(-0.05, 0.05)
    shoulder_y_diff = rng.uniform(0, 5, n)
    shoulder_width  = rng.normal(1.7, 0.15, n).clip(1.2, 2.2)   # wide, back straight
    forward_lean    = rng.normal(3.2, 0.25, n).clip(2.4, 4.2)   # head properly above shoulders
    hip_y_diff      = rng.uniform(0, 5, n)
    torso_tilt      = rng.normal(0.0, 0.05, n).clip(-0.1, 0.1)
    torso_length    = rng.normal(2.1, 0.2, n).clip(1.6, 2.8)    # tall posture

    X_good = np.column_stack([
        shoulder_angle, shoulder_y_diff, shoulder_width,
        forward_lean, hip_y_diff, torso_tilt, torso_length
    ]).astype(np.float32)
    X_parts.append(X_good)
    y_parts.append(np.zeros(n, dtype=np.int32))

    # ----- CLASS 1: FAIR POSTURE -----
    n = n_per_class
    shoulder_angle  = rng.uniform(-0.15, 0.15, n)
    shoulder_y_diff = rng.uniform(5, 22, n)
    shoulder_width  = rng.normal(1.4, 0.2, n).clip(1.0, 2.0)
    forward_lean    = rng.uniform(1.5, 2.4, n)                  # mild forward head
    hip_y_diff      = rng.uniform(5, 18, n)
    torso_tilt      = rng.uniform(-0.35, 0.35, n)
    torso_length    = rng.normal(1.7, 0.2, n).clip(1.2, 2.2)

    X_fair = np.column_stack([
        shoulder_angle, shoulder_y_diff, shoulder_width,
        forward_lean, hip_y_diff, torso_tilt, torso_length
    ]).astype(np.float32)
    X_parts.append(X_fair)
    y_parts.append(np.ones(n, dtype=np.int32))

    # ----- CLASS 2: BAD POSTURE -----
    n = n_per_class
    # Mix of slouch-sideways and forward-head cases
    n_sideways    = n // 2
    n_forward     = n - n_sideways

    # Sideways slouch
    shoulder_angle_s  = rng.uniform(0.15, 0.50, n_sideways) * rng.choice([-1, 1], n_sideways)
    shoulder_y_diff_s = rng.uniform(22, 60, n_sideways)
    shoulder_width_s  = rng.normal(1.2, 0.2, n_sideways).clip(0.8, 1.6)
    forward_lean_s    = rng.normal(2.5, 0.4, n_sideways).clip(1.0, 3.5)
    hip_y_diff_s      = rng.uniform(18, 45, n_sideways)
    torso_tilt_s      = rng.uniform(0.4, 1.0, n_sideways) * rng.choice([-1, 1], n_sideways)
    torso_length_s    = rng.normal(1.3, 0.2, n_sideways).clip(0.8, 1.8)

    # Severe forward head
    shoulder_angle_f  = rng.normal(0.0, 0.05, n_forward).clip(-0.1, 0.1)
    shoulder_y_diff_f = rng.uniform(0, 15, n_forward)
    shoulder_width_f  = rng.normal(1.3, 0.2, n_forward).clip(0.9, 1.8)
    forward_lean_f    = rng.uniform(0.2, 1.4, n_forward)        # head very far forward
    hip_y_diff_f      = rng.uniform(0, 12, n_forward)
    torso_tilt_f      = rng.normal(0.0, 0.1, n_forward).clip(-0.2, 0.2)
    torso_length_f    = rng.normal(1.2, 0.15, n_forward).clip(0.8, 1.6)

    X_bad = np.vstack([
        np.column_stack([shoulder_angle_s, shoulder_y_diff_s, shoulder_width_s,
                         forward_lean_s, hip_y_diff_s, torso_tilt_s, torso_length_s]),
        np.column_stack([shoulder_angle_f, shoulder_y_diff_f, shoulder_width_f,
                         forward_lean_f, hip_y_diff_f, torso_tilt_f, torso_length_f]),
    ]).astype(np.float32)
    X_parts.append(X_bad)
    y_parts.append(np.full(n, 2, dtype=np.int32))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
