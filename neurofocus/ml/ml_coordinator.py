"""
ML Coordinator — orchestrates online learning for fatigue and posture modules.
Manages warm-up, computes ML blend weights, exposes calibration progress.
"""

import time

from .user_profile import UserProfile
from .threshold_adapter import ThresholdAdapter


class MLCoordinator:
    """Coordinates online learning for all ML modules."""

    ML_WARMUP_DURATION = 60.0   # seconds of data collection
    BLEND_RAMP_DURATION = 120.0  # seconds for geometry->ML progression

    def __init__(self, fatigue_classifier=None, posture_classifier=None,
                 profile_id: str = "default"):
        self.fc = fatigue_classifier
        self.pc = posture_classifier

        # Load or create user profile
        self.user_profile = UserProfile.load(profile_id)
        self.threshold_adapter = ThresholdAdapter(self.user_profile)

        # Online learner — background fine-tuning of ML models
        from .online_learner import OnlineLearner
        self.online_learner = OnlineLearner(
            model_type="fatigue_lstm",
            model=getattr(self.fc, 'lstm_model', None) or getattr(self.fc, 'model', None),
        )

        self.warmup_started = False
        self.ml_warmup_complete = False
        self.ml_warmup_start: float = 0.0
        self._blend_active = False

    # -- warm-up --

    def start_warmup(self):
        if not self.warmup_started:
            self.threshold_adapter.start_warmup()
            self.ml_warmup_start = time.time()
            self.warmup_started = True

    def update(self, ear: float, mar: float, pitch: float, current_time: float,
               face_is_visible: bool = True):
        """
        Feed a measurement into the warm-up pipeline.

        Parameters:
            ear, mar, pitch   — current sensor values
            current_time      — wall-clock timestamp
            face_is_visible   — set False to PAUSE warm-up accumulation
        """
        if not self.warmup_started:
            self.start_warmup()
        self.threshold_adapter.add_sample(ear, mar, pitch, current_time,
                                          face_is_visible=face_is_visible)

        # Transition to ML mode
        if (not self.ml_warmup_complete
                and self.threshold_adapter.completed):
            self._on_warmup_complete()

    def _on_warmup_complete(self):
        self.ml_warmup_complete = True
        self._blend_active = True
        self._enable_ml_models()

        # Refresh online learner model reference — LSTM model is now
        # guaranteed to be loaded
        if self.fc is not None and self.online_learner is not None:
            lstm = getattr(self.fc, 'lstm_model', None) or getattr(self.fc, 'model', None)
            if lstm is not None:
                self.online_learner.model = lstm
                print(f"[MLCoordinator] Online learner wired to LSTM model "
                      f"(layers: {len(lstm.layers)}).")

        print("[MLCoordinator] Warm-up complete — personalized thresholds active, ML enabled.")

    def _enable_ml_models(self):
        if self.fc is not None:
            self.fc.set_thresholds(self.threshold_adapter)
            self.fc.user_profile = self.user_profile
        if self.pc is not None:
            self.pc.enable_ml_progressive()

    # -- blend weight --

    def get_ml_blend_weight(self) -> float:
        """0.0 = pure geometric, 1.0 = pure ML."""
        if not self.ml_warmup_complete:
            return 0.0
        elapsed = time.time() - self.ml_warmup_start
        return min(1.0, elapsed / self.BLEND_RAMP_DURATION)

    def get_calibration_progress(self) -> int:
        """0–100 for UI progress bar."""
        if self.ml_warmup_complete:
            return 100
        if not self.ml_warmup_start:
            return 0
        return int(min(100.0, self.threshold_adapter.progress * 100))

    # -- online learning --

    def collect_sample(self, ear: float, mar: float, pitch: float,
                       fatigue_level: str, current_time: float):
        """
        Add a labeled feature vector to the online-learning buffer.

        Called from the main video loop every N frames.  The actual
        fine-tuning happens in a background thread when enough samples
        have accumulated.

        Parameters:
            ear, mar, pitch    — current sensor values
            fatigue_level      — 'normal', 'mild', 'moderate', 'severe'
                                 (used as a pseudo-label)
            current_time       — wall-clock timestamp
        """
        if self.online_learner is None or self.online_learner.model is None:
            return

        import numpy as np

        # Build a 16-D feature vector matching TemporalFeatureExtractor output
        # (simplified: use current values as proxies for rolling-window stats)
        feature_vector = np.array([
            ear,                          # ear_mean
            0.01,                         # ear_std  (proxy)
            max(0.1, ear - 0.02),         # ear_min
            min(0.4, ear + 0.02),         # ear_max
            0.0,                          # ear_trend  (needs history — default 0)
            mar,                          # mar_mean
            mar,                          # mar_max
            0.0,                          # mar_trend
            1.0 if mar > 0.5 else 0.0,    # yawning
            max(0.0, mar - 0.3),          # yawn_intensity
            0.0001,                       # ear_variance_long
            0.9,                          # ear_stability (assume stable)
            15.0,                         # estimated_blink_rate (proxy)
            max(0, pitch * 0.01),         # head_droop_mean
            abs(pitch) * 0.01,            # head_tilt_mean
            0.5,                          # time_elapsed (proxy)
        ], dtype=np.float32)

        # Pseudo-label from fatigue level (used as weak supervision)
        label_map = {'normal': 0, 'mild': 1, 'moderate': 1, 'severe': 2}
        label = label_map.get(fatigue_level.lower(), None)

        self.online_learner.add_sample(feature_vector, label)

        # Check if we should retrain
        if self.online_learner.can_retrain():
            started = self.online_learner.start_retrain()
            if started:
                print(f"[MLCoordinator] Online retrain started "
                      f"(buffer: {self.online_learner.buffer_size} samples)")
