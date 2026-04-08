"""
Blink tracker using EAR signal thresholding.
Detects blink events from Eye Aspect Ratio changes over time.
Thresholds can be adapted per-user via ThresholdAdapter.
"""

import time
from collections import deque

OPEN_DEFAULT = 0.28
CLOSED_DEFAULT = 0.22

class BlinkTracker:
    """Real-time blink tracking based on EAR threshold crossings."""

    def __init__(self, open_threshold: float = OPEN_DEFAULT,
                 closed_threshold: float = CLOSED_DEFAULT):
        self.open_threshold = open_threshold
        self.closed_threshold = closed_threshold
        self.last_ear = 0.35
        self.was_closed = False
        self.blink_timestamps: deque = deque(maxlen=60)

    def update(self, ear: float, current_time: float):
        closed = ear < self.closed_threshold
        is_open = ear > self.open_threshold
        if not self.was_closed and closed:
            self.blink_timestamps.append(current_time)
            self.was_closed = True
        elif is_open:
            self.was_closed = False
        self.last_ear = ear

    def set_thresholds(self, open_thresh: float, closed_thresh: float):
        self.open_threshold = open_thresh
        self.closed_threshold = closed_thresh

    def get_blink_rate_per_minute(self, window: float = 60.0) -> int:
        now = time.time()
        recent = [t for t in self.blink_timestamps if now - t < window]
        if len(recent) < 2:
            return 0
        span = recent[-1] - recent[0]
        if span < 1.0:
            span = 1.0
        return int(len(recent) / span * 60)
