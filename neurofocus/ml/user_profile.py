"""
Per-user baseline storage and personalization.
Persists to data/user_profiles/{profile_id}.json between sessions.
"""

import json
import os


class UserProfile:
    """Stores per-user biometric baselines and provides personalized thresholds."""

    def __init__(self, profile_id: str = "default",
                 data_dir: str = "data/user_profiles"):
        self.profile_id = profile_id
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # EAR: personal open / closed ranges
        self.ear_open_mean = 0.30
        self.ear_open_std = 0.03
        self.ear_closed_mean = 0.18
        self.ear_closed_std = 0.02

        # MAR: personal yawn threshold
        self.mar_normal_mean = 0.15
        self.mar_yawn_threshold = 0.60

        # Posture: personal head-pitch baseline
        self.pose_head_pitch_mean = 0.0
        self.pose_head_pitch_std = 5.0

        self.calibrated = False

    def apply_personalization(self, ear: float, mar: float, blink_rate: int) -> dict:
        """Return adjustment factors for fatigue score calculation."""
        ear_range = max(self.ear_open_mean - self.ear_closed_mean, 0.01)
        ear_normalized = (ear - self.ear_closed_mean) / ear_range

        if ear_normalized < 0.3:
            personal_fatigue_factor = 1.2
        elif ear_normalized > 0.7:
            personal_fatigue_factor = 0.8
        else:
            personal_fatigue_factor = 1.0

        return {"personal_fatigue_factor": personal_fatigue_factor}

    def get_personalized_ear_closed_threshold(self) -> float:
        gap = self.ear_open_mean - self.ear_closed_mean
        self.ear_closed_mean + gap * 0.3
        return self.ear_closed_mean + gap * 0.3

    def get_personalized_mar_yawn_threshold(self) -> float:
        return self.mar_yawn_threshold

    def save(self):
        path = os.path.join(self.data_dir, f"{self.profile_id}.json")
        data = {
            "ear_open_mean": self.ear_open_mean,
            "ear_open_std": self.ear_open_std,
            "ear_closed_mean": self.ear_closed_mean,
            "ear_closed_std": self.ear_closed_std,
            "mar_normal_mean": self.mar_normal_mean,
            "mar_yawn_threshold": self.mar_yawn_threshold,
            "pose_head_pitch_mean": self.pose_head_pitch_mean,
            "pose_head_pitch_std": self.pose_head_pitch_std,
            "calibrated": self.calibrated,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"Warning: could not save user profile to {path}: {e}")

    @classmethod
    def load(cls, profile_id: str = "default",
             data_dir: str = "data/user_profiles") -> "UserProfile":
        profile = cls(profile_id, data_dir)
        path = os.path.join(data_dir, f"{profile_id}.json")
        if not os.path.exists(path):
            return profile

        try:
            with open(path, "r") as f:
                data = json.load(f)
            profile.ear_open_mean = data.get("ear_open_mean", profile.ear_open_mean)
            profile.ear_open_std = data.get("ear_open_std", profile.ear_open_std)
            profile.ear_closed_mean = data.get("ear_closed_mean", profile.ear_closed_mean)
            profile.ear_closed_std = data.get("ear_closed_std", profile.ear_closed_std)
            profile.mar_normal_mean = data.get("mar_normal_mean", profile.mar_normal_mean)
            profile.mar_yawn_threshold = data.get("mar_yawn_threshold", profile.mar_yawn_threshold)
            profile.pose_head_pitch_mean = data.get("pose_head_pitch_mean", profile.pose_head_pitch_mean)
            profile.pose_head_pitch_std = data.get("pose_head_pitch_std", profile.pose_head_pitch_std)
            profile.calibrated = data.get("calibrated", False)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: could not load user profile from {path}: {e}")

        return profile
