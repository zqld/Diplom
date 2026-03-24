"""
Face Detector - MediaPipe Face Landmarker wrapper.
"""

import cv2
import numpy as np
import os


mp = None
mediapipe_available = False
FaceLandmarker = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.core.base_options import BaseOptions
    mediapipe_available = True
except (ImportError, Exception) as e:
    print(f"MediaPipe not available: {e}")


class FaceDetector:
    """Face detector using MediaPipe Face Landmarker."""
    
    def __init__(self, max_faces: int = 1, min_confidence: float = 0.8):
        self._detector = None
        self.available = False
        self.max_faces = max_faces
        
        if not mediapipe_available:
            return
            
        try:
            model_path = 'models/face_landmarker.task'
            if not os.path.exists(model_path):
                print(f"FaceLandmarker model not found: {model_path}")
                return
                
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_faces=max_faces,
                min_face_detection_confidence=min_confidence,
                min_tracking_confidence=0.8
            )
            self._detector = FaceLandmarker.create_from_options(options)
            self.available = True
        except Exception as e:
            print(f"FaceDetector init error: {e}")
    
    @property
    def detector(self):
        return self._detector
    
    @property
    def is_available(self) -> bool:
        return self.available
    
    def process_frame(self, frame, draw: bool = True):
        """Process frame and detect faces.
        
        Args:
            frame: BGR frame from OpenCV
            draw: Whether to draw landmarks
            
        Returns:
            annotated_frame, results
        """
        if self._detector is None:
            return frame, None
            
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            result = self._detector.detect(mp_image)
            
            annotated_frame = frame.copy()
            
            if draw and result.face_landmarks:
                self._draw_face_mesh(annotated_frame, result.face_landmarks)
            
            return annotated_frame, result
            
        except Exception as e:
            print(f"FaceDetector process error: {e}")
            return frame, None
    
    def _draw_face_mesh(self, image, face_landmarks_list):
        """Draw face mesh landmarks."""
        try:
            h, w = image.shape[:2]
            
            for face_landmarks in face_landmarks_list:
                for landmark in face_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        except Exception:
            pass
    
    def get_landmarks(self, results):
        """Extract face landmarks from results."""
        if results is None:
            return None
        if hasattr(results, 'face_landmarks') and results.face_landmarks:
            return results.face_landmarks[0]
        return None
    
    def is_face_detected(self, results) -> bool:
        """Check if any face was detected."""
        return results and hasattr(results, 'face_landmarks') and len(results.face_landmarks) > 0
