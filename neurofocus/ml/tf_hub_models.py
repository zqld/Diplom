"""
TensorFlow Hub Models for NeuroFocus.
Provides pre-trained models for feature extraction and classification.
"""

import os
import numpy as np
import cv2
import tensorflow as tf


class TFHubPoseEstimator:
    """
    Pose estimation using MoveNet from TensorFlow Hub.
    Provides faster and potentially more accurate pose detection than MediaPipe.
    """
    
    def __init__(self, model_url: str = None):
        """
        Initialize MoveNet pose estimator.
        
        Args:
            model_url: TF Hub model URL. Options:
                - 'https://tfhub.dev/google/movenet/singlepose/lightning/4' (fastest)
                - 'https://tfhub.dev/google/movenet/singlepose/thunder/4' (more accurate)
        """
        self.model = None
        self.model_url = model_url or 'https://tfhub.dev/google/movenet/singlepose/lightning/4'
        self._load_model()
        
        # Keypoint names for MoveNet (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Keypoint indices
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.NOSE = 0
    
    def _load_model(self):
        """Load MoveNet model from TensorFlow Hub."""
        try:
            import tensorflow_hub as hub
            print(f"Loading MoveNet from {self.model_url}...")
            self.model = hub.load(self.model_url)
            print("MoveNet loaded successfully!")
        except Exception as e:
            print(f"Failed to load MoveNet: {e}")
            self.model = None
    
    @property
    def is_available(self) -> bool:
        return self.model is not None
    
    def estimate(self, frame):
        """
        Estimate pose in a frame.
        
        Args:
            frame: BGR image from OpenCV (HxWx3)
        
        Returns:
            numpy array of shape (17, 3) with keypoints [x, y, score], or None
        """
        if self.model is None:
            return None
        
        try:
            # Convert BGR to RGB and resize to expected input size (192x192)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (192, 192))
            img = tf.convert_to_tensor(resized, dtype=tf.int32)
            img = tf.expand_dims(img, 0)
            
            # Run inference
            model = self.model.signatures['serving_default']
            outputs = model(img)
            
            # Get keypoints (shape: [1, 1, 17, 3])
            keypoints = outputs['output_0'].numpy()[0, 0]
            
            return keypoints
            
        except Exception as e:
            print(f"Pose estimation error: {e}")
            return None
    
    def estimate_from_image(self, image_path: str):
        """Load image from file and estimate pose."""
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        return self.estimate(frame)
    
    def get_shoulder_angle(self, keypoints):
        """Calculate shoulder angle in degrees."""
        if keypoints is None:
            return None
        
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        
        # Calculate angle
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle
    
    def get_shoulder_level_diff(self, keypoints):
        """Get difference in shoulder heights (asymmetry)."""
        if keypoints is None:
            return None
        
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        
        return abs(left_shoulder[1] - right_shoulder[1]) * 100  # Convert to percentage
    
    def get_pose_features(self, keypoints):
        """
        Extract pose features for classification.
        
        Returns:
            dict with features for posture classification
        """
        if keypoints is None:
            return None
        
        # Shoulder features
        shoulder_angle = self.get_shoulder_angle(keypoints)
        shoulder_diff = self.get_shoulder_level_diff(keypoints)
        
        # Hip features
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        hip_diff = abs(left_hip[1] - right_hip[1]) * 100
        
        # Torso alignment
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        torso_tilt = (shoulder_center_x - hip_center_x) * 100
        
        # Forward lean
        nose = keypoints[self.NOSE]
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        forward_lean = shoulder_center_y - nose[1]
        
        return {
            'shoulder_angle': shoulder_angle,
            'shoulder_diff': shoulder_diff,
            'hip_diff': hip_diff,
            'torso_tilt': torso_tilt,
            'forward_lean': forward_lean,
            'confidence': np.mean([k[2] for k in keypoints])  # Average confidence
        }
    
    def draw_pose(self, frame, keypoints, thickness=2):
        """Draw pose skeleton on frame."""
        if keypoints is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Connections for drawing
        connections = [
            (0, 1), (0, 2),  # nose to eyes
            (1, 3), (2, 4),   # eyes to ears
            (5, 6),           # shoulders
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12), # shoulders to hips
            (11, 12),          # hips
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16)  # right leg
        ]
        
        # Draw connections
        for idx1, idx2 in connections:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                x1, y1 = int(keypoints[idx1][0] * w), int(keypoints[idx1][1] * h)
                x2, y2 = int(keypoints[idx2][0] * w), int(keypoints[idx2][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        
        # Draw keypoints
        for keypoint in keypoints:
            x, y = int(keypoint[0] * w), int(keypoint[1] * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        return frame


class TFHubFaceDetector:
    """
    Face detection using BlazeFace from TensorFlow Hub.
    Lightweight and fast face detector.
    
    Note: MediaPipe Face Landmarker is recommended for NeuroFocus
    as it provides more detailed landmarks (478 points).
    This class is kept for future enhancements.
    """
    
    def __init__(self, model_url: str = None):
        """
        Initialize BlazeFace detector.
        
        Args:
            model_url: TF Hub model URL for face detection
        """
        self.model = None
        # Try different BlazeFace model URLs
        self.model_url = model_url or 'https://tfhub.dev/google/face-detection/1'
        self._load_model()
    
    def _load_model(self):
        """Load face detection model from TensorFlow Hub."""
        try:
            import tensorflow_hub as hub
            print(f"Loading face detector from {self.model_url}...")
            self.model = hub.load(self.model_url)
            print("Face detector loaded successfully!")
        except Exception as e:
            print(f"Failed to load face detector: {e}")
            # Fall back to MediaPipe which is already working
            print("Note: MediaPipe Face Landmarker will be used instead")
            self.model = None
    
    @property
    def is_available(self) -> bool:
        return self.model is not None
    
    def detect(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            list of bounding boxes [y_min, x_min, y_max, x_max] normalized, or empty list
        """
        if self.model is None:
            return []
        
        try:
            # Preprocess
            img = tf.cast(frame, tf.string)
            
            # Run inference
            model = self.model.signatures['serving_default']
            outputs = model(img)
            
            # Get boxes (format varies by model)
            if 'boxes' in outputs:
                boxes = outputs['boxes'].numpy()
                return boxes.tolist()
            elif 'detection_boxes' in outputs:
                boxes = outputs['detection_boxes'].numpy()
                return boxes.tolist()
            else:
                return []
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []


def download_models_for_offline():
    """
    Download TensorFlow Hub models for offline use.
    Models will be cached in ~/.cache/tf-hub/
    """
    print("Downloading TensorFlow Hub models for offline use...")
    
    models = [
        ('MoveNet Lightning', 'https://tfhub.dev/google/movenet/singlepose/lightning/4'),
        ('MoveNet Thunder', 'https://tfhub.dev/google/movenet/singlepose/thunder/4'),
    ]
    
    for name, url in models:
        try:
            print(f"Downloading {name}...")
            import tensorflow_hub as hub
            model = hub.load(url)
            print(f"  {name} downloaded and cached!")
        except Exception as e:
            print(f"  Failed to download {name}: {e}")
    
    print("Done! Models are cached for offline use.")


if __name__ == '__main__':
    # Test model loading
    print("Testing TensorFlow Hub models...\n")
    
    print("1. Testing MoveNet...")
    pose_estimator = TFHubPoseEstimator()
    print(f"   Available: {pose_estimator.is_available}\n")
    
    print("2. Testing BlazeFace...")
    face_detector = TFHubFaceDetector()
    print(f"   Available: {face_detector.is_available}\n")
    
    if pose_estimator.is_available:
        # Test on sample image
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                keypoints = pose_estimator.estimate(frame)
                if keypoints is not None:
                    features = pose_estimator.get_pose_features(keypoints)
                    print(f"Pose features: {features}")
        cap.release()
    
    print("\nTo download models for offline use, run:")
    print("  python -c 'from neurofocus.ml.tf_hub_models import download_models_for_offline; download_models_for_offline()'")