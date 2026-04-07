"""
Posture Classifier using ML.
Predicts posture quality from pose landmarks.
Supports both MediaPipe and TensorFlow Hub MoveNet models.
"""

import numpy as np
from collections import deque


class PostureClassifier:
    """
    ML-based posture classifier using TensorFlow.
    Uses pose landmarks (especially shoulders) for classification.
    
    Classes: good, fair, bad
    
    Can use:
    - MediaPipe Pose landmarks (default)
    - TensorFlow Hub MoveNet (optional, more accurate)
    """
    
    def __init__(self, model_path: str = None, use_tf_hub: bool = False):
        self.model = None
        self.model_path = model_path
        self.classes = ['good', 'fair', 'bad']
        
        # Use TF Hub MoveNet for better accuracy
        self._use_tf_hub = use_tf_hub
        self._tf_hub_estimator = None
        
        # Fallback to geometric features if model not available
        self._use_fallback = True
        
        # History for smoothing
        self.prediction_history = deque(maxlen=10)
        
        # Geometric thresholds (fallback)
        self._shoulder_tilt_threshold = 12  # degrees
        self._shoulder_diff_threshold = 15   # pixels (normalized)
        self._forward_lean_threshold = 0.08
        
        # Load models
        if use_tf_hub:
            self._init_tf_hub()
        
        if model_path:
            self._load_model(model_path)
    
    def _init_tf_hub(self):
        """Initialize TensorFlow Hub MoveNet estimator."""
        try:
            from .tf_hub_models import TFHubPoseEstimator
            self._tf_hub_estimator = TFHubPoseEstimator()
            print(f"TF Hub MoveNet initialized: {self._tf_hub_estimator.is_available}")
        except Exception as e:
            print(f"Failed to init TF Hub: {e}")
            self._tf_hub_estimator = None
    
    def _load_model(self, model_path: str):
        """Load TensorFlow/Keras model."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self._use_fallback = False
            print(f"Posture model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load posture model: {e}")
            self._use_fallback = True
    
    def predict(self, pose_landmarks):
        """
        Predict posture quality from pose landmarks.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks (33 points)
        
        Returns:
            dict with 'status', 'confidence', 'raw_scores'
        """
        if pose_landmarks is None or len(pose_landmarks) < 13:
            return {
                'status': 'unknown',
                'confidence': 0.0,
                'raw_scores': [0.0, 0.0, 0.0]
            }
        
        # Use ML model if available
        if not self._use_fallback:
            return self._predict_ml(pose_landmarks)
        
        # Use geometric fallback
        return self._predict_geometric(pose_landmarks)
    
    def predict_from_frame(self, frame):
        """
        Predict posture from a video frame using TensorFlow Hub MoveNet.
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            dict with 'status', 'confidence', 'raw_scores'
        """
        if self._tf_hub_estimator is None or not self._tf_hub_estimator.is_available:
            return {
                'status': 'unknown',
                'confidence': 0.0,
                'raw_scores': [0.0, 0.0, 0.0]
            }
        
        try:
            # Run MoveNet inference
            keypoints = self._tf_hub_estimator.estimate(frame)
            
            if keypoints is None:
                return {
                    'status': 'unknown',
                    'confidence': 0.0,
                    'raw_scores': [0.0, 0.0, 0.0]
                }
            
            # Extract features
            features = self._tf_hub_estimator.get_pose_features(keypoints)
            
            # Classify based on features
            shoulder_angle = abs(features['shoulder_angle'])
            shoulder_diff = features['shoulder_diff']
            forward_lean = abs(features['forward_lean'])
            
            scores = [0.0, 0.0, 0.0]  # [good, fair, bad]
            confidence = 0.0
            
            # Classification based on thresholds
            bad_score = 0
            
            # Check shoulder tilt
            if shoulder_angle > self._shoulder_tilt_threshold:
                bad_score += 40
            elif shoulder_angle > 8:
                scores[0] -= 20
            
            # Check shoulder asymmetry
            if shoulder_diff > self._shoulder_diff_threshold:
                bad_score += 30
            elif shoulder_diff > 10:
                scores[0] -= 15
            
            # Check forward lean
            if forward_lean > self._forward_lean_threshold:
                bad_score += 30
            elif forward_lean > 0.05:
                scores[0] -= 15
            
            # Determine final status
            if bad_score >= 50:
                status = 'bad'
                confidence = min(1.0, bad_score / 100)
                scores[2] = 0.8
                scores[1] = 0.15
            elif bad_score >= 25:
                status = 'fair'
                confidence = min(1.0, bad_score / 60)
                scores[1] = 0.7
                scores[0] = 0.2
            else:
                status = 'good'
                confidence = 0.85
                scores[0] = 0.85
                scores[1] = 0.1
            
            # Apply smoothing
            self.prediction_history.append(status)
            smoothed_status = self._smooth_prediction()
            
            return {
                'status': smoothed_status,
                'confidence': confidence,
                'raw_scores': scores,
                'shoulder_tilt': shoulder_angle,
                'shoulder_diff': shoulder_diff,
                'forward_lean': forward_lean
            }
            
        except Exception as e:
            print(f"Error in TF Hub posture prediction: {e}")
            return {
                'status': 'unknown',
                'confidence': 0.0,
                'raw_scores': [0.0, 0.0, 0.0]
            }
    
    def _predict_geometric(self, pose_landmarks):
        """
        Геометрическая классификация осанки по плечам и положению головы.

        Работает в нормализованных координатах (0-1) MediaPipe Pose:
          nose        = landmark 0
          left_ear    = landmark 7
          right_ear   = landmark 8
          left_shoulder  = landmark 11
          right_shoulder = landmark 12

        Критерии (все в нормализованных единицах, Y растёт вниз):
          head_height  — расстояние носа над центром плеч по Y.
                         Норма: 0.20–0.40. Меньше → голова опустилась / ссутулился.
          shoulder_diff — |left_shoulder.y − right_shoulder.y|.
                         Норма: < 0.05. Больше → один локоть поднят, тело наклонено.
          shoulder_angle — угол линии плеч в градусах.
                         Норма: < 8°. Больше → видимый наклон корпуса.
        """
        import math

        try:
            # Получаем список landmarks
            if hasattr(pose_landmarks, 'landmark'):
                lms = pose_landmarks.landmark
            elif isinstance(pose_landmarks, list) and pose_landmarks and hasattr(pose_landmarks[0], 'x'):
                lms = pose_landmarks
            else:
                return {'status': 'unknown', 'confidence': 0.0, 'raw_scores': [0.0, 0.0, 0.0]}

            if len(lms) < 13:
                return {'status': 'unknown', 'confidence': 0.0, 'raw_scores': [0.0, 0.0, 0.0]}

            nose       = (lms[0].x,  lms[0].y)
            l_shoulder = (lms[11].x, lms[11].y)
            r_shoulder = (lms[12].x, lms[12].y)

            # ── 1. Высота головы над плечами ────────────────────────────────
            shoulder_cy = (l_shoulder[1] + r_shoulder[1]) / 2.0
            head_height = shoulder_cy - nose[1]   # > 0 когда нос выше плеч

            # ── 2. Асимметрия плеч по вертикали ────────────────────────────
            shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])

            # ── 3. Угол наклона линии плеч ──────────────────────────────────
            dx = r_shoulder[0] - l_shoulder[0]
            dy = r_shoulder[1] - l_shoulder[1]
            shoulder_angle = abs(math.degrees(math.atan2(dy, dx + 1e-6)))
            # atan2 может вернуть угол > 90° для почти вертикальных линий;
            # берём эквивалентный острый угол
            if shoulder_angle > 90:
                shoulder_angle = 180 - shoulder_angle

            # ── Подсчёт баллов ──────────────────────────────────────────────
            bad_score = 0

            # Голова слишком близко к плечам (ссутулился / откинулся назад)
            if head_height < 0.15:
                bad_score += 45
            elif head_height < 0.22:
                bad_score += 20

            # Асимметрия плеч
            if shoulder_diff > 0.08:
                bad_score += 35
            elif shoulder_diff > 0.045:
                bad_score += 15

            # Наклон корпуса
            if shoulder_angle > 12:
                bad_score += 20
            elif shoulder_angle > 7:
                bad_score += 8

            # ── Итог ────────────────────────────────────────────────────────
            if bad_score >= 45:
                status = 'bad'
                confidence = min(0.92, bad_score / 100.0)
            elif bad_score >= 22:
                status = 'fair'
                confidence = 0.65
            else:
                status = 'good'
                confidence = 0.88

            self.prediction_history.append(status)
            smoothed_status = self._smooth_prediction()

            return {
                'status': smoothed_status,
                'confidence': confidence,
                'raw_scores': [0.0, 0.0, 0.0],
                'shoulder_angle': shoulder_angle,
                'shoulder_diff': shoulder_diff,
                'head_height': head_height,
            }

        except Exception as e:
            print(f"Error in geometric posture prediction: {e}")
            return {'status': 'unknown', 'confidence': 0.0, 'raw_scores': [0.0, 0.0, 0.0]}
    
    def _predict_ml(self, pose_landmarks):
        """Use ML model for prediction."""
        try:
            from .preprocessing import extract_pose_features
            
            features = extract_pose_features(pose_landmarks)
            
            if features is None:
                return self._predict_geometric(pose_landmarks)
            
            # Prepare input
            feature_input = np.expand_dims(features, axis=0)
            
            # Predict
            preds = self.model.predict(feature_input, verbose=0)[0]
            
            status = self.classes[np.argmax(preds)]
            confidence = float(preds[np.argmax(preds)])
            
            self.prediction_history.append(status)
            smoothed_status = self._smooth_prediction()
            
            return {
                'status': smoothed_status,
                'confidence': confidence,
                'raw_scores': preds.tolist()
            }
            
        except Exception as e:
            print(f"Error in ML posture prediction: {e}")
            return self._predict_geometric(pose_landmarks)
    
    def _smooth_prediction(self):
        """Apply smoothing to predictions using voting."""
        if len(self.prediction_history) < 3:
            return self.prediction_history[-1] if self.prediction_history else 'unknown'
        
        from collections import Counter
        counts = Counter(self.prediction_history)
        return counts.most_common(1)[0][0]
    
    def predict_from_face_mesh(self, face_landmarks, frame_width=640, frame_height=480):
        """
        Когда плечи не видны (pose landmarks недоступны) — возвращаем 'unknown',
        чтобы не генерировать ложные тревоги осанки.
        Полноценная оценка возможна только при видимых плечах.
        """
        return {'status': 'unknown', 'confidence': 0.0}

    def _predict_from_face_mesh_impl(self, face_landmarks, frame_width=640, frame_height=480):
        """
        (Отключён) Predict posture from face mesh (fallback when pose not available).
        Uses face position and orientation as proxy for posture.

        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width, frame_height: frame dimensions
        
        Returns:
            dict with 'status', 'confidence'
        """
        if face_landmarks is None:
            return {
                'status': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Handle both list and object formats
            if hasattr(face_landmarks, 'landmark'):
                landmarks = face_landmarks.landmark
            elif isinstance(face_landmarks, list):
                landmarks = face_landmarks
            else:
                return {
                    'status': 'unknown',
                    'confidence': 0.0
                }
            
            # Helper to get x, y from landmark
            def get_coords(idx):
                lm = landmarks[idx]
                if isinstance(lm, dict):
                    return (lm.get('x', 0.5), lm.get('y', 0.5))
                elif hasattr(lm, 'x'):
                    return (lm.x, lm.y)
                elif isinstance(lm, (list, tuple)):
                    return (lm[0], lm[1])
                return (0.5, 0.5)
            
            # Extract features from face
            nose_x, nose_y = get_coords(1)
            left_ear_x, left_ear_y = get_coords(234)
            right_ear_x, right_ear_y = get_coords(454)
            
            # Face center position
            face_center_x = (left_ear_x + right_ear_x) / 2
            face_center_y = (nose_y + left_ear_y + right_ear_y) / 3
            
            # Calculate deviation from center
            x_deviation = abs(face_center_x - 0.5)
            y_deviation = abs(face_center_y - 0.4)
            
            # Head tilt from ears
            head_tilt = abs(right_ear_x - left_ear_x)
            
            # Simple classification
            if x_deviation > 0.15 or y_deviation > 0.2:
                status = 'bad'
                confidence = 0.7
            elif x_deviation > 0.08 or y_deviation > 0.12:
                status = 'fair'
                confidence = 0.6
            else:
                status = 'good'
                confidence = 0.8
            
            self.prediction_history.append(status)
            smoothed_status = self._smooth_prediction()
            
            return {
                'status': smoothed_status,
                'confidence': confidence,
                'x_deviation': x_deviation,
                'y_deviation': y_deviation,
                'head_tilt': head_tilt
            }
            
        except Exception as e:
            print(f"Error in face-based posture prediction: {e}")
            return {
                'status': 'unknown',
                'confidence': 0.0
            }
    
    def train(self, X_train, y_train, epochs: int = 50, save_path: str = None):
        """
        Train the classifier on user data.
        
        Args:
            X_train: Feature array
            y_train: Labels (0=good, 1=fair, 2=bad)
            epochs: Training epochs
            save_path: Path to save trained model
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Build simple model
            model = keras.Sequential([
                layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
                layers.Dense(16, activation='relu'),
                layers.Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train
            model.fit(X_train, y_train, epochs=epochs, verbose=1)
            
            self.model = model
            self._use_fallback = False
            
            if save_path:
                model.save(save_path)
                print(f"Model saved to {save_path}")
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def get_status_text(self, status: str) -> str:
        """Get human-readable status text."""
        status_map = {
            'good': 'Хорошая',
            'fair': 'Средняя',
            'bad': 'Плохая',
            'unknown': 'Анализ...'
        }
        return status_map.get(status, 'Анализ...')