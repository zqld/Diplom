"""
ML Classifiers для NeuroFocus.
Используются из gui_main.py для классификации усталости и осанки.
"""

from .fatigue_classifier import FatigueClassifier
from .posture_classifier import PostureClassifier
from .preprocessing import (
    extract_eye_region,
    extract_pose_features,
    prepare_face_image,
    calculate_ear_from_landmarks,
    calculate_mar_from_landmarks,
)

# Online learning modules
from .blink_tracker import BlinkTracker
from .microsleep_detector import MicrosleepDetector
from .temporal_features import TemporalFeatureExtractor
from .user_profile import UserProfile
from .threshold_adapter import ThresholdAdapter
from .online_learner import OnlineLearner
from .ml_coordinator import MLCoordinator

__all__ = [
    "FatigueClassifier",
    "PostureClassifier",
    "extract_eye_region",
    "extract_pose_features",
    "prepare_face_image",
    "calculate_ear_from_landmarks",
    "calculate_mar_from_landmarks",
    # Online learning
    "BlinkTracker",
    "MicrosleepDetector",
    "TemporalFeatureExtractor",
    "UserProfile",
    "ThresholdAdapter",
    "OnlineLearner",
    "MLCoordinator",
]
