"""
Hand Detector - MediaPipe Hand Landmarker wrapper.
"""

import cv2
import numpy as np
import os


mp = None
mediapipe_available = False

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.core.base_options import BaseOptions
    mediapipe_available = True
except (ImportError, Exception) as e:
    print(f"MediaPipe not available: {e}")


class HandDetector:
    """Hand detector using MediaPipe Hand Landmarker."""
    
    def __init__(self, max_hands: int = 1, min_confidence: float = 0.7):
        self._detector = None
        self.available = False
        self.max_hands = max_hands
        
        if not mediapipe_available:
            print("HandDetector: MediaPipe not available, gestures disabled")
            return
            
        try:
            model_path = 'models/hand_landmarker.task'
            if not os.path.exists(model_path):
                print(f"HandLandmarker model not found: {model_path}")
                return
                
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_hands=max_hands,
                min_hand_detection_confidence=min_confidence,
                min_tracking_confidence=0.5
            )
            self._detector = HandLandmarker.create_from_options(options)
            self.available = True
        except Exception as e:
            print(f"HandDetector init error: {e}")
    
    @property
    def detector(self):
        return self._detector
    
    @property
    def is_available(self) -> bool:
        return self.available
    
    def process_frame(self, frame, draw: bool = True):
        """Process frame and detect hands."""
        if not self.available:
            return frame, None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = self._detector.detect(mp_image)
            
            if draw and results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    self._draw_hand(frame, hand_landmarks)
            
            return frame, results
            
        except Exception as e:
            print(f"HandDetector process error: {e}")
            return frame, None
    
    def _draw_hand(self, image, hand_landmarks):
        """Draw hand landmarks."""
        try:
            h, w = image.shape[:2]
            
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            
            for idx1, idx2 in connections:
                if idx1 < len(hand_landmarks) and idx2 < len(hand_landmarks):
                    x1 = int(hand_landmarks[idx1].x * w)
                    y1 = int(hand_landmarks[idx1].y * h)
                    x2 = int(hand_landmarks[idx2].x * w)
                    y2 = int(hand_landmarks[idx2].y * h)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception:
            pass
    
    def get_landmarks(self, results):
        """Extract hand landmarks from results."""
        if results is None:
            return None
        if hasattr(results, 'hand_landmarks') and results.hand_landmarks:
            return results.hand_landmarks
        return None
    
    def is_hand_detected(self, results) -> bool:
        """Check if any hand was detected."""
        return results and hasattr(results, 'hand_landmarks') and len(results.hand_landmarks) > 0
    
    def get_fingers_up(self, landmarks):
        """Get which fingers are up."""
        if not landmarks:
            return [False] * 5
        
        fingers = []
        lm = landmarks
        
        if lm[4].x > lm[3].x:
            fingers.append(True)
        else:
            fingers.append(False)
        
        for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if lm[tip_idx].y < lm[pip_idx].y:
                fingers.append(True)
            else:
                fingers.append(False)
        
        return fingers
    
    def get_index_tip(self, landmarks, frame_width, frame_height):
        """Get index finger tip position."""
        if not landmarks:
            return None
        index_tip = landmarks[8]
        return (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
