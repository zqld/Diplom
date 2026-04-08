"""
Temporal feature extractor — maintains rolling windows of EAR, MAR, and head pose.
Computes mean, std, min, max, trend for LSTM input and trend analysis.
"""

import numpy as np
from collections import deque


class TemporalFeatureExtractor:
    """Rolling-window feature extraction for temporal analysis."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.ear_history: deque = deque(maxlen=window_size)
        self.mar_history: deque = deque(maxlen=window_size)
        self.head_droop_history: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)

    def update(self, ear: float, mar: float, head_droop: float = 0.0,
               head_tilt: float = 0.0, current_time: float = None) -> dict:
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.head_droop_history.append(head_droop)
        if current_time is not None:
            self.timestamps.append(current_time)
        return self._compute_features(head_tilt)

    def _compute_features(self, head_tilt: float) -> dict:
        ear_arr = np.array(self.ear_history) if self.ear_history else np.array([0.3])
        mar_arr = np.array(self.mar_history) if self.mar_history else np.array([0.15])

        n = len(ear_arr)
        if n >= 15:
            ear_trend = float(np.mean(ear_arr[-7:]) - np.mean(ear_arr[:7]))
            ear_variance_long = float(np.var(ear_arr))
        else:
            ear_trend = 0.0
            ear_variance_long = 0.0

        ear_stability = float(1.0 - min(1.0, np.std(ear_arr) * 10)) if n >= 5 else 1.0

        # Blink rate estimation from zero-crossings below closed threshold
        ear_closed_thresh = 0.22
        blink_count = 0
        for i in range(1, n):
            if ear_arr[i - 1] > ear_closed_thresh and ear_arr[i] <= ear_closed_thresh:
                blink_count += 1
        if n >= 5:
            # Estimate per-minute rate assuming ~30 FPS sampling
            window_duration = n / 30.0  # seconds
            estimated_blink_rate = int(blink_count / max(window_duration, 1.0) * 60)
        else:
            estimated_blink_rate = 0

        # MAR features
        if n >= 10:
            mar_trend = float(np.mean(mar_arr[-5:]) - np.mean(mar_arr[:5]))
        else:
            mar_trend = 0.0

        recent_mar = mar_arr[-5:] if n >= 5 else mar_arr
        yawn_intensity = float(np.max(recent_mar)) if len(recent_mar) > 0 else 0.0
        yawning = bool(np.max(recent_mar) > 0.5) if len(recent_mar) > 0 else False
        mar_max = float(np.max(mar_arr)) if len(mar_arr) > 0 else 0.15

        return {
            'ear_mean': float(np.mean(ear_arr)),
            'ear_std': float(np.std(ear_arr)),
            'ear_min': float(np.min(ear_arr)),
            'ear_max': float(np.max(ear_arr)),
            'ear_trend': ear_trend,
            'mar_mean': float(np.mean(mar_arr)),
            'mar_max': mar_max,
            'mar_trend': mar_trend,
            'yawning': yawning,
            'yawn_intensity': yawn_intensity,
            'ear_variance_long': ear_variance_long,
            'ear_stability': ear_stability,
            'estimated_blink_rate': estimated_blink_rate,
        }
