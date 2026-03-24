"""
Pose Detector - MediaPipe Pose Landmarker wrapper.
"""

import cv2
import numpy as np
import os


mp = None
mediapipe_available = False

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
    from mediapipe.tasks.python.core.base_options import BaseOptions
    mediapipe_available = True
except (ImportError, Exception) as e:
    print(f"MediaPipe not available: {e}")


class PoseDetector:
    """Pose detector using MediaPipe Pose Landmarker."""
    
    def __init__(self, model_path: str = 'models/pose_landmarker_lite.task'):
        self._detector = None
        self.available = False
        self.model_path = model_path
        
        if not mediapipe_available:
            return
            
        try:
            if not os.path.exists(model_path):
                print(f"PoseLandmarker model not found: {model_path}")
                return
                
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
            )
            self._detector = PoseLandmarker.create_from_options(options)
            self.available = True
        except Exception as e:
            print(f"PoseDetector init error: {e}")
    
    @property
    def detector(self):
        return self._detector
    
    @property
    def is_available(self) -> bool:
        return self.available
    
    def process_frame(self, frame, draw: bool = True):
        """Process frame and detect pose."""
        if self._detector is None:
            return frame, None
            
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            result = self._detector.detect(mp_image)
            
            annotated_frame = frame.copy()
            
            if draw and result.pose_landmarks:
                self._draw_pose(annotated_frame, result.pose_landmarks)
            
            return annotated_frame, result
            
        except Exception as e:
            print(f"PoseDetector process error: {e}")
            return frame, None
    
    def _draw_pose(self, image, pose_landmarks):
        """Draw pose landmarks."""
        try:
            h, w = image.shape[:2]
            
            for landmark in pose_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            
            connections_list = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24)
            ]
            
            for idx1, idx2 in connections_list:
                if idx1 < len(pose_landmarks) and idx2 < len(pose_landmarks):
                    x1 = int(pose_landmarks[idx1].x * w)
                    y1 = int(pose_landmarks[idx1].y * h)
                    x2 = int(pose_landmarks[idx2].x * w)
                    y2 = int(pose_landmarks[idx2].y * h)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception:
            pass
    
    def get_landmarks(self, results):
        """Extract pose landmarks from results."""
        if results is None:
            return None
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            return results.pose_landmarks[0]
        return None
    
    def is_pose_detected(self, results) -> bool:
        """Check if pose was detected."""
        return results and hasattr(results, 'pose_landmarks') and len(results.pose_landmarks) > 0
    
    def get_shoulder_coords(self, landmarks):
        """Get shoulder coordinates."""
        if not landmarks or len(landmarks) < 13:
            return None, None
            
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        return (left_shoulder.x, left_shoulder.y), (right_shoulder.x, right_shoulder.y)
