"""
Online threshold adaptation through warm-up data collection.

Collects EAR / MAR / pitch samples **only when the face is visible**.
If the face is lost, the warm-up timer is paused and resumes when the
face returns — preventing corrupted personalised thresholds.

After enough valid samples, computes personalised thresholds via
bimodal clustering and saves them to the user profile.
"""

import time
import numpy as np
from collections import deque


class ThresholdAdapter:
    """
    Adapt ML thresholds to the individual user during a warm-up phase.

    Key features:
    • Only accumulates samples when face_is_visible == True.
    • Timer is PAUSED while face is lost, RESUMED on reappearance.
    • Requires WARMUP_DURATION seconds of VALID (face-visible) data.
    """

    WARMUP_DURATION = 60.0   # seconds of VALID data
    MIN_SAMPLES = 100        # minimum ear samples needed

    def __init__(self, user_profile):
        self.profile = user_profile

        # State
        self.warmup_start: float | None = None     # real-world start time
        self.valid_accumulated: float = 0.0         # seconds of valid data
        self.last_valid_timestamp: float | None = None
        self._paused = False

        # Buffers
        self.ear_samples: deque = deque(maxlen=5_000)
        self.mar_samples: deque = deque(maxlen=5_000)
        self.pitch_samples: deque = deque(maxlen=5_000)

        # Flags
        self.active = False
        self.completed = False

        # Progress 0–1 for UI
        self.progress = 0.0

    # ── Public API ─────────────────────────────────────────────

    def start_warmup(self, current_time: float | None = None):
        """Begin warm-up.  Call once before the first add_sample()."""
        if not self.active:
            t = current_time if current_time is not None else time.time()
            self.warmup_start = t
            self.valid_accumulated = 0.0
            self.last_valid_timestamp = t
            self.active = True
            self.completed = False
            self._paused = False

    def add_sample(self, ear: float, mar: float, pitch: float,
                   current_time: float,
                   face_is_visible: bool = True):
        """
        Add a measurement.  Only counted when face_is_visible.

        Parameters:
            ear, mar, pitch  — current sensor values
            current_time     — wall-clock timestamp
            face_is_visible  — set False if face was not detected
        """
        if not self.active or self.completed:
            return

        now = current_time

        if face_is_visible:
            # ── Resume from pause ──────────────────────────
            if self._paused:
                # Shift warmup_start forward by the pause duration
                pause_duration = now - self.last_valid_timestamp
                self.warmup_start += pause_duration
                self._paused = False

            # Accumulate
            self.ear_samples.append(ear)
            self.mar_samples.append(mar)
            self.pitch_samples.append(pitch)

            elapsed = now - self.warmup_start
            self.valid_accumulated = elapsed
            self.last_valid_timestamp = now
            self.progress = min(1.0, elapsed / self.WARMUP_DURATION)

            # Check completion
            if (elapsed >= self.WARMUP_DURATION
                    and len(self.ear_samples) >= self.MIN_SAMPLES):
                self._compute_personalized_thresholds()
                self.active = False
                self.completed = True
                self.progress = 1.0
        else:
            # ── Face lost — pause ──────────────────────────
            if not self._paused:
                self._paused = True
                # progress bar freezes at current value

    def add_valid_sample(self, ear: float, mar: float, pitch: float,
                         current_time: float):
        """Convenience: explicitly mark sample as valid (face visible)."""
        self.add_sample(ear, mar, pitch, current_time, face_is_visible=True)

    def add_invalid_sample(self, current_time: float):
        """Convenience: explicitly mark sample as invalid (face lost)."""
        self.add_sample(0.0, 0.0, 0.0, current_time, face_is_visible=False)

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def remaining_seconds(self) -> float:
        """Estimated seconds of valid data still needed."""
        if self.completed:
            return 0.0
        return max(0.0, self.WARMUP_DURATION - self.valid_accumulated)

    # ── Internals ──────────────────────────────────────────────

    def _compute_personalized_thresholds(self):
        """
        Compute personalised EAR / MAR / pitch thresholds from valid samples.

        Heuristic: sort EAR values; lower 30 % ≈ closed-eyes, upper 70 % ≈ open.
        """
        ear_arr = np.array(sorted(self.ear_samples))
        mar_arr = np.array(self.mar_samples)
        pitch_arr = np.array(self.pitch_samples)
        n = len(ear_arr)

        # Bimodal split
        split = int(n * 0.3)
        closed_part = ear_arr[:split]
        open_part = ear_arr[split:]

        self.profile.ear_closed_mean = float(np.mean(closed_part))
        self.profile.ear_closed_std = float(np.std(closed_part))
        self.profile.ear_open_mean = float(np.mean(open_part))
        self.profile.ear_open_std = float(np.std(open_part))

        self.profile.mar_normal_mean = float(np.median(mar_arr))
        self.profile.mar_yawn_threshold = float(np.clip(
            np.percentile(mar_arr, 90), 0.35, 1.0))

        self.profile.pose_head_pitch_mean = float(np.mean(pitch_arr))
        self.profile.pose_head_pitch_std = float(np.std(pitch_arr))
        self.profile.calibrated = True
        self.profile.save()
