"""
Preprocessing utilities for ML classifiers.
Extract features from MediaPipe landmarks and images.
"""

import cv2
import numpy as np


def extract_eye_region(frame, face_landmarks, target_size=(64, 64)):
    """
    Extract eye region from face frame for fatigue detection.
    
    Args:
        frame: BGR image from OpenCV
        face_landmarks: MediaPipe face landmarks
        target_size: (width, height) for output
    
    Returns:
        resized grayscale image of eye region, or None
    """
    if face_landmarks is None:
        return None
    
    try:
        h, w = frame.shape[:2]
        
        # Eye landmarks indices for both eyes
        # Left eye: 33, 133, 160, 144, 158, 153
        # Right eye: 362, 263, 385, 380, 387, 373
        
        # Handle both list and object formats
        if hasattr(face_landmarks, 'landmark'):
            landmarks = face_landmarks.landmark
        elif isinstance(face_landmarks, list):
            landmarks = face_landmarks
        else:
            return None
        
        def get_eye_bbox(indices):
            x_coords = [landmarks[i].x * w for i in indices]
            y_coords = [landmarks[i].y * h for i in indices]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            pad_x = int((x_max - x_min) * 0.3)
            pad_y = int((y_max - y_min) * 0.3)
            
            x_min = max(0, x_min - pad_x)
            x_max = min(w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h, y_max + pad_y)
            
            return x_min, y_min, x_max, y_max
        
        # Get bounding boxes for both eyes
        left_eye_bbox = get_eye_bbox([33, 133, 160, 144, 158, 153])
        right_eye_bbox = get_eye_bbox([362, 263, 385, 380, 387, 373])
        
        # Combine both eyes into one region
        x_min = min(left_eye_bbox[0], right_eye_bbox[0])
        y_min = min(left_eye_bbox[1], right_eye_bbox[1])
        x_max = max(left_eye_bbox[2], right_eye_bbox[2])
        y_max = max(left_eye_bbox[3], right_eye_bbox[3])
        
        # Extract region
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return None
        
        # Convert to grayscale and resize
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_resized = cv2.resize(eye_gray, target_size)
        
        # Normalize to [0, 1]
        eye_normalized = eye_resized.astype(np.float32) / 255.0
        
        return eye_normalized
        
    except Exception as e:
        print(f"Error extracting eye region: {e}")
        return None


def extract_pose_features(pose_landmarks):
    """
    Extract pose features for posture classification.
    
    Supports:
    - MediaPipe Pose result object (has .landmark)
    - MediaPipe landmark list (each has .x, .y attributes) - indices 0-32
    - MoveNet numpy array (shape: 17x3) - indices 0,5,6,11,12
    - None or invalid input
    
    Args:
        pose_landmarks: pose data in any of the above formats
    
    Returns:
        numpy array of 7 features, or None
    """
    if pose_landmarks is None:
        return None
    
    # Determine the type and extract points
    try:
        # Case 1: MediaPipe result object with .landmark attribute
        if hasattr(pose_landmarks, 'landmark'):
            landmarks = pose_landmarks.landmark
            if len(landmarks) < 25:
                return None
            # MediaPipe indices
            MP_INDICES = {'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12, 'left_hip': 23, 'right_hip': 24}
            get_point = lambda key: (landmarks[MP_INDICES[key]].x, landmarks[MP_INDICES[key]].y)
        
        # Case 2: List of landmark objects (MediaPipe format)
        elif isinstance(pose_landmarks, list):
            if len(pose_landmarks) < 1:
                return None
            # Check if it's a list of landmark objects with .x, .y
            if hasattr(pose_landmarks[0], 'x') and hasattr(pose_landmarks[0], 'y'):
                if len(pose_landmarks) < 25:
                    return None
                # MediaPipe indices for list
                MP_INDICES = {'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12, 'left_hip': 23, 'right_hip': 24}
                get_point = lambda key: (pose_landmarks[MP_INDICES[key]].x, pose_landmarks[MP_INDICES[key]].y)
            # Check if it's MoveNet format (list of lists/tuples)
            elif isinstance(pose_landmarks[0], (list, tuple, np.ndarray)):
                arr = np.array(pose_landmarks)
                if arr.shape[0] < 13:
                    return None
                # MoveNet indices
                MV_INDICES = {'nose': 0, 'left_shoulder': 5, 'right_shoulder': 6, 'left_hip': 11, 'right_hip': 12}
                get_point = lambda key: (float(arr[MV_INDICES[key]][0]), float(arr[MV_INDICES[key]][1]))
            else:
                return None
        
        # Case 3: NumPy array (MoveNet format)
        elif isinstance(pose_landmarks, np.ndarray):
            arr = pose_landmarks
            if arr.shape[0] < 13:
                return None
            MV_INDICES = {'nose': 0, 'left_shoulder': 5, 'right_shoulder': 6, 'left_hip': 11, 'right_hip': 12}
            get_point = lambda key: (float(arr[MV_INDICES[key]][0]), float(arr[MV_INDICES[key]][1]))
        
        else:
            return None
            
    except Exception as e:
        print(f"Pose feature extraction type error: {e}")
        return None
    
    try:
        features = []
        scale = 640.0
        
        # Get key points
        left_shoulder = get_point('left_shoulder')
        right_shoulder = get_point('right_shoulder')
        
        # Shoulder angle
        shoulder_dy = (right_shoulder[1] - left_shoulder[1]) * scale
        shoulder_dx = (right_shoulder[0] - left_shoulder[0]) * scale
        shoulder_angle = np.arctan2(shoulder_dy, shoulder_dx + 1e-6)
        features.append(shoulder_angle)
        
        # Shoulder symmetry
        shoulder_y_diff = abs(left_shoulder[1] - right_shoulder[1]) * scale
        features.append(shoulder_y_diff)
        
        # Shoulder width
        shoulder_width = np.sqrt(
            ((right_shoulder[0] - left_shoulder[0]) * scale) ** 2 +
            ((right_shoulder[1] - left_shoulder[1]) * scale) ** 2
        )
        features.append(shoulder_width / 100.0)
        
        # Forward lean
        nose = get_point('nose')
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        forward_lean = (shoulder_center_y - nose[1]) * scale
        features.append(forward_lean / 50.0)
        
        # Hip positions
        left_hip = get_point('left_hip')
        right_hip = get_point('right_hip')
        
        # Hip symmetry
        hip_y_diff = abs(left_hip[1] - right_hip[1]) * scale
        features.append(hip_y_diff)
        
        # Torso tilt
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        torso_tilt = (shoulder_center_x - hip_center_x) * scale
        features.append(torso_tilt / 50.0)
        
        # Torso length
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        torso_length = abs(hip_center_y - shoulder_center_y) * scale
        features.append(torso_length / 100.0)
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Pose feature calculation error: {e}")
        return None


def prepare_face_image(frame, face_landmarks, target_size=(64, 64)):
    """
    Prepare face image for classification.
    
    Args:
        frame: BGR image from OpenCV
        face_landmarks: MediaPipe face landmarks
        target_size: (width, height) for output
    
    Returns:
        resized grayscale image, or None
    """
    if face_landmarks is None:
        return None
    
    try:
        h, w = frame.shape[:2]
        
        # Handle both list and object formats
        if hasattr(face_landmarks, 'landmark'):
            landmarks = face_landmarks.landmark
        elif isinstance(face_landmarks, list):
            landmarks = face_landmarks
        else:
            return None
        
        # Get face bounding box from landmarks
        x_coords = [p.x for p in landmarks]
        y_coords = [p.y for p in landmarks]
        
        min_x, max_x = int(min(x_coords) * w), int(max(x_coords) * w)
        min_y, max_y = int(min(y_coords) * h), int(max(y_coords) * h)
        
        # Add padding
        pad_x = int((max_x - min_x) * 0.15)
        pad_y = int((max_y - min_y) * 0.15)
        
        min_x = max(0, min_x - pad_x)
        min_y = max(0, min_y - pad_y)
        max_x = min(w, max_x + pad_x)
        max_y = min(h, max_y + pad_y)
        
        # Extract and process
        face_region = frame[min_y:max_y, min_x:max_x]
        
        if face_region.size == 0:
            return None
        
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, target_size)
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        return face_normalized
        
    except Exception as e:
        print(f"Error preparing face image: {e}")
        return None


def calculate_ear_from_landmarks(face_landmarks):
    """
    Calculate Eye Aspect Ratio from face landmarks.
    
    Args:
        face_landmarks: MediaPipe face landmarks
    
    Returns:
        EAR value, or None
    """
    if face_landmarks is None:
        return None
    
    try:
        # Handle both list and object formats
        if hasattr(face_landmarks, 'landmark'):
            landmarks = face_landmarks.landmark
        elif isinstance(face_landmarks, list):
            landmarks = face_landmarks
        else:
            return None
        
        # Helper to get x, y from landmark (handles dict, object, or array)
        def get_coords(idx):
            lm = landmarks[idx]
            if isinstance(lm, dict):
                return np.array([lm.get('x', 0), lm.get('y', 0)])
            elif hasattr(lm, 'x'):
                return np.array([lm.x, lm.y])
            elif isinstance(lm, (list, tuple, np.ndarray)):
                return np.array([lm[0], lm[1]])
            return np.array([0, 0])
        
        LEFT_EYE_IDXS = [33, 133, 160, 144, 158, 153]
        RIGHT_EYE_IDXS = [362, 263, 385, 380, 387, 373]
        
        def eye_aspect_ratio(indices):
            p1 = get_coords(indices[0])
            p4 = get_coords(indices[1])
            p2 = get_coords(indices[2])
            p6 = get_coords(indices[3])
            p3 = get_coords(indices[4])
            p5 = get_coords(indices[5])
            
            vertical_1 = np.linalg.norm(p2 - p6)
            vertical_2 = np.linalg.norm(p3 - p5)
            horizontal = np.linalg.norm(p1 - p4)
            
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
            return ear
        
        left_ear = eye_aspect_ratio(LEFT_EYE_IDXS)
        right_ear = eye_aspect_ratio(RIGHT_EYE_IDXS)
        
        return (left_ear + right_ear) / 2.0
        
    except Exception as e:
        print(f"Error calculating EAR: {e}")
        return None


def calculate_mar_from_landmarks(face_landmarks):
    """
    Calculate Mouth Aspect Ratio from face landmarks.
    
    Args:
        face_landmarks: MediaPipe face landmarks
    
    Returns:
        MAR value, or None
    """
    if face_landmarks is None:
        return None
    
    try:
        # Handle both list and object formats
        if hasattr(face_landmarks, 'landmark'):
            landmarks = face_landmarks.landmark
        elif isinstance(face_landmarks, list):
            landmarks = face_landmarks
        else:
            return None
        
        # Helper to get x, y from landmark (handles dict, object, or array)
        def get_coords(idx):
            lm = landmarks[idx]
            if isinstance(lm, dict):
                return np.array([lm.get('x', 0), lm.get('y', 0)])
            elif hasattr(lm, 'x'):
                return np.array([lm.x, lm.y])
            elif isinstance(lm, (list, tuple, np.ndarray)):
                return np.array([lm[0], lm[1]])
            return np.array([0, 0])
        
        MOUTH_IDXS = [61, 291, 13, 14]
        
        p_left = get_coords(MOUTH_IDXS[0])
        p_right = get_coords(MOUTH_IDXS[1])
        p_top = get_coords(MOUTH_IDXS[2])
        p_bottom = get_coords(MOUTH_IDXS[3])
        
        vertical = np.linalg.norm(p_top - p_bottom)
        horizontal = np.linalg.norm(p_left - p_right)
        
        mar = vertical / (horizontal + 1e-6)
        return mar
        
    except Exception as e:
        print(f"Error calculating MAR: {e}")
        return None