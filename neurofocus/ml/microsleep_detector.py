"""
Microsleep detector — tracks prolonged eye-closure events.
A microsleep is defined as eyes closed below threshold for >= min_duration seconds.
Thresholds are adapted per-user via ThresholdAdapter.
"""

import time
from collections import deque

THRESHOLD_DEFAULT = 0.20

class MicrosleepDetector:
    """Detect microsleep events from sustained low EAR values."""

    def __init__(self, threshold: float = THRESHOLD_DEFAULT,
                 min_duration: float = 1.5):
        self.threshold = threshold
        self.min_duration = min_duration
        self._eyes_closed_since: float | None = None
        self.microsleep_times: deque = deque(maxlen=30)

    def update(self, ear: float, current_time: float):
        t = self.threshold
        if ear < t:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = current_time
        else:
            if self._eyes_closed_since is not None:
                duration = current_time - self._eyes_closed_since
                if duration >= self.min_duration:
                    self.microsleep_times.append(current_time)
            self._eyes_closed_since = None

    def set_threshold(self, value: float):
        self.threshold = value

    def get_statistics(self) -> dict:
        now = time.time()
        recent = [t for t in self.microsleep_times if now - t < 60.0]
        count = len(recent)
        if count >= 3:
            danger = 'danger'
        elif count >= 1:
            danger = 'warning'
        else:
            danger = 'safe'
        return {
            'microsleeps_per_minute': count,
            'danger_level': danger,
            'currently_closed': self._eyes_closed_since is not None,
        }
