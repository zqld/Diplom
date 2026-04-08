"""
Online learner — accumulates labeled feature vectors and periodically
fine-tunes the top layers of the LSTM / Dense model.

LSTM models require temporal input (window_size × features).  The learner
maintains a rolling buffer of individual frames and assembles them into
overlapping windows when enough data is available.

Runs retraining off the main thread so it does not block the video loop.
"""

import time
import threading
import numpy as np
from collections import deque

RETRAIN_THRESHOLD = 500    # labeled samples needed
RETRAIN_COOLDOWN = 600     # seconds between fine-tuning runs (10 min)
WINDOW_SIZE = 30           # LSTM temporal window


class OnlineLearner:
    """
    Collect samples, periodically fine-tune model top layers in background.

    Works with both LSTM (temporal) and Dense (flat) models:
    - LSTM: assembles windows of WINDOW_SIZE consecutive samples
    - Dense: trains on individual feature vectors directly
    """

    def __init__(self, model_type: str = "fatigue_lstm", model=None,
                 window_size: int = WINDOW_SIZE):
        self.model_type = model_type
        self.model = model
        self.window_size = window_size

        # Raw sample buffer (individual frames)
        self._raw_buffer: deque = deque(maxlen=5_000)
        # Labeled windows ready for training
        self._labeled_windows: list = []
        # Unlabeled windows (accumulated between retrains)
        self._pending_windows: list = []

        self.last_retrain_time = 0.0
        self.is_training = False
        self._lock = threading.Lock()

    def add_sample(self, feature_vector: np.ndarray, label: int | None = None):
        """
        Add a single-frame feature vector.

        Windows are assembled automatically when enough consecutive frames
        with the same (non-None) label are available.
        """
        with self._lock:
            self._raw_buffer.append((feature_vector, label))

            # Try to assemble a new window ending at this sample
            self._try_assemble_window()

    def _try_assemble_window(self):
        """
        If the last `window_size` consecutive samples all have the same
        non-None label, create a training window.
        """
        if len(self._raw_buffer) < self.window_size:
            return

        # Look at the tail of the buffer
        tail = list(self._raw_buffer)[-self.window_size:]

        # All labels must be non-None and identical
        labels = [lbl for _, lbl in tail]
        if any(lbl is None for lbl in labels):
            return

        first_label = labels[0]
        if not all(lbl == first_label for lbl in labels):
            return

        # Assemble window: (window_size, features)
        features = np.array([f for f, _ in tail], dtype=np.float32)
        self._labeled_windows.append((features, first_label))

    def ready_to_retrain(self) -> bool:
        with self._lock:
            return len(self._labeled_windows) >= RETRAIN_THRESHOLD

    def can_retrain(self) -> bool:
        return (self.ready_to_retrain()
                and time.time() - self.last_retrain_time > RETRAIN_COOLDOWN
                and self.model is not None
                and not self.is_training)

    def start_retrain(self):
        """Start fine-tuning in a background thread — does not block caller."""
        if not self.can_retrain():
            return False
        t = threading.Thread(target=self._retrain_loop, daemon=True)
        t.start()
        return True

    def _retrain_loop(self):
        """Actual fine-tuning logic, called from background thread."""
        self.is_training = True
        try:
            import tensorflow as tf

            with self._lock:
                windows = self._labeled_windows[:RETRAIN_THRESHOLD]
                self._labeled_windows = self._labeled_windows[RETRAIN_THRESHOLD:]

            if len(windows) < RETRAIN_THRESHOLD:
                print("[OnlineLearner] Not enough windows after re-lock, skipping.")
                return

            X = np.array([w[0] for w in windows], dtype=np.float32)
            y = np.array([w[1] for w in windows], dtype=np.int32)

            # Freeze all but last 2 trainable layers
            trainable_count = 0
            for layer in reversed(self.model.layers):
                if trainable_count < 2:
                    layer.trainable = True
                else:
                    layer.trainable = False
                trainable_count += 1

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='sparse_categorical_crossentropy',
            )
            self.model.fit(X, y, epochs=5, batch_size=32, verbose=0)

            # Unfreeze everything for future use
            for layer in self.model.layers:
                layer.trainable = True

            self.last_retrain_time = time.time()
            print(f"[OnlineLearner] Fine-tuned {self.model_type} "
                  f"on {X.shape[0]} windows ({X.shape}).")
        except Exception as e:
            print(f"[OnlineLearner] Retrain error: {e}")
        finally:
            self.is_training = False

    @property
    def buffer_size(self):
        with self._lock:
            return len(self._labeled_windows)

    @property
    def raw_buffer_size(self):
        with self._lock:
            return len(self._raw_buffer)
