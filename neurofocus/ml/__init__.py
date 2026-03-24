"""
ML Classifiers for NeuroFocus.
Provides fatigue and posture classification using TensorFlow.

Components:
- FatigueClassifier: TensorFlow CNN for drowsiness detection (awake/drowsy/sleeping)
- PostureClassifier: TensorFlow Hub MoveNet for posture quality (good/fair/bad)
- TFHubPoseEstimator: MoveNet pose estimation wrapper
- TrainingDataCollector: Collects user data for model retraining
- ModelTrainer: Trains custom models on collected data
- BlinkTracker: Real-time blink detection from EAR
- MicrosleepDetector: Detects microsleep episodes
- TemporalFeatureExtractor: Extracts temporal features for LSTM
- LSTMFatigueModel: LSTM-based temporal fatigue analysis
- LSTMPostureModel: LSTM-based temporal posture analysis
- UserProfileManager: User profiles with calibration and personalization
- FatigueAutoLabeler: Auto-labels fatigue data based on behavior
- PostureAutoLabeler: Auto-labels posture data based on behavior

Usage:
    from neurofocus.ml import FatigueClassifier, PostureClassifier
    
    fc = FatigueClassifier()  # TensorFlow CNN + LSTM
    pc = PostureClassifier(use_tf_hub=True)  # MoveNet + LSTM
    
    fatigue = fc.predict(face_landmarks, frame)
    posture = pc.predict(pose_landmarks)
"""

from .fatigue_classifier import FatigueClassifier
from .posture_classifier import PostureClassifier
from .preprocessing import (
    extract_eye_region,
    extract_pose_features,
    prepare_face_image,
    calculate_ear_from_landmarks,
    calculate_mar_from_landmarks
)
from .training_data import TrainingDataCollector
from .model_trainer import ModelTrainer
from .tf_hub_models import (
    TFHubPoseEstimator,
    download_models_for_offline
)
from .blink_tracker import BlinkTracker, BlinkDetector
from .microsleep_detector import MicrosleepDetector
from .temporal_features import TemporalFeatureExtractor, TemporalFeatures
from .lstm_fatigue_model import LSTMFatigueModel, LSTMClassifier
from .lstm_posture_model import LSTMPostureModel
from .user_profile import UserProfileManager
from .auto_labeler import FatigueAutoLabeler, PostureAutoLabeler

__all__ = [
    # Classifiers
    'FatigueClassifier',
    'PostureClassifier',
    
    # Preprocessing
    'extract_eye_region',
    'extract_pose_features',
    'prepare_face_image',
    'calculate_ear_from_landmarks',
    'calculate_mar_from_landmarks',
    
    # Data collection & training
    'TrainingDataCollector',
    'ModelTrainer',
    
    # TensorFlow Hub
    'TFHubPoseEstimator',
    'download_models_for_offline',
    
    # Advanced features
    'BlinkTracker',
    'BlinkDetector',
    'MicrosleepDetector',
    'TemporalFeatureExtractor',
    'TemporalFeatures',
    'LSTMFatigueModel',
    'LSTMClassifier',
    'LSTMPostureModel',
    'UserProfileManager',
    'FatigueAutoLabeler',
    'PostureAutoLabeler',
]