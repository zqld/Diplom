"""
NeuroFocus — модули машинного обучения и детекторы.
Используются из gui_main.py для ML-классификации осанки и усталости.
"""

__version__ = "1.0.0"

# Detectors (используется PoseDetector из gui_main.py)
from neurofocus.detectors.pose_detector import PoseDetector

# ML (используются из gui_main.py)
from neurofocus.ml.fatigue_classifier import FatigueClassifier
from neurofocus.ml.posture_classifier import PostureClassifier
from neurofocus.ml.preprocessing import extract_pose_features

__all__ = [
    "PoseDetector",
    "FatigueClassifier",
    "PostureClassifier",
    "extract_pose_features",
]
