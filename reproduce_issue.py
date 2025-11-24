import cv2
import numpy as np
from src.pose_estimator import HeadPoseEstimator

def test_pose_estimator():
    estimator = HeadPoseEstimator()
    
    # Mock frame dimensions
    img_w = 640
    img_h = 480
    frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    # Create mock landmarks for a straight face
    # MediaPipe landmarks are normalized [0, 1]
    # We need to match the indices used in pose_estimator.py:
    # [1, 152, 33, 263, 61, 291]
    # 1: Nose tip (Center)
    # 152: Chin (Below nose)
    # 33: Left eye (Left of nose, above)
    # 263: Right eye (Right of nose, above)
    # 61: Left mouth corner (Left, below nose)
    # 291: Right mouth corner (Right, below nose)
    
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class MockLandmarks:
        def __init__(self):
            self.landmark = [MockLandmark(0, 0)] * 468 # Initialize with dummy
            
            # Center of face
            cx, cy = 0.5, 0.5
            
            # Scale factors (approximate)
            dx = 0.1
            dy = 0.1
            
            # Set specific landmarks
            # Nose
            self.landmark[1] = MockLandmark(cx, cy)
            # Chin (lower Y in image is higher value)
            self.landmark[152] = MockLandmark(cx, cy + dy * 1.5)
            # Left Eye
            self.landmark[33] = MockLandmark(cx - dx, cy - dy)
            # Right Eye
            self.landmark[263] = MockLandmark(cx + dx, cy - dy)
            # Left Mouth
            self.landmark[61] = MockLandmark(cx - dx * 0.5, cy + dy * 0.5)
            # Right Mouth
            self.landmark[291] = MockLandmark(cx + dx * 0.5, cy + dy * 0.5)

    landmarks = MockLandmarks()
    
    print("Testing with current model points...")
    pitch, yaw, roll = estimator.get_pose(frame, landmarks)
    print(f"Raw Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")
    
    # Now let's try with inverted Y model points
    print("\nTesting with inverted Y model points...")
    original_model_points = estimator.model_points.copy()
    
    # Invert Y
    new_model_points = original_model_points.copy()
    new_model_points[:, 1] = -new_model_points[:, 1]
    
    estimator.model_points = new_model_points
    pitch_new, yaw_new, roll_new = estimator.get_pose(frame, landmarks)
    print(f"New Pitch: {pitch_new:.2f}, Yaw: {yaw_new:.2f}, Roll: {roll_new:.2f}")

if __name__ == "__main__":
    test_pose_estimator()
