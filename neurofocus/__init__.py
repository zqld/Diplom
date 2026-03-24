"""
NeuroFocus - система мониторинга усталости и осанки.
"""

__version__ = "1.0.0"

# Detectors
from neurofocus.detectors.face_detector import FaceDetector
from neurofocus.detectors.hand_detector import HandDetector
from neurofocus.detectors.pose_detector import PoseDetector

# Analyzers (Emotion only - Fatigue and Posture use ML classifiers from neurofocus.ml)
from neurofocus.analyzers.emotion import EmotionDetector as EmotionAnalyzer

# Utils
from neurofocus.utils.landmarks import LandmarkUtils, landmarks_utils
from neurofocus.utils.geometry import calculate_ear, calculate_mar
from neurofocus.utils.pose_estimator import HeadPoseEstimator

# Controls
from neurofocus.controls.gesture import GestureController
from neurofocus.controls.calibration import CalibrationManager

# Core
from neurofocus.core.config import config_manager
from neurofocus.utils.logger import logger
from neurofocus.utils.database import DatabaseManager
from neurofocus.utils.sound import sound_manager

__all__ = [
    # Detectors
    "FaceDetector",
    "HandDetector",
    "PoseDetector",
    # Analyzers
    "EmotionAnalyzer",
    # Utils
    "LandmarkHelper",
    "landmarks_utils",
    "calculate_ear",
    "calculate_mar",
    "HeadPoseEstimator",
    # Controls
    "GestureController",
    "CalibrationManager",
    # Core
    "config_manager",
    "logger",
    "DatabaseManager",
    "sound_manager",
]
