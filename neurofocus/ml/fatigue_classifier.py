"""
Fatigue Classifier using TensorFlow ML.
Predicts drowsiness/fatigue from face images using CNN.
Fully TensorFlow-based solution with temporal analysis and personalization.
"""

import numpy as np
from collections import deque
import os
import time


class FatigueClassifier:
    """
    TensorFlow CNN-based fatigue classifier with temporal analysis.
    Uses eye region images for drowsiness detection + temporal features.
    
    Classes: awake, drowsy, sleeping
    
    Features:
    - CNN-based classification
    - Real blink rate tracking
    - Microsleep detection
    - Temporal feature extraction
    - User personalization
    """
    
    def __init__(self, model_path: str = None, user_profile_manager=None):
        self.model = None
        self.model_path = model_path or 'models/fatigue_cnn.keras'
        self.classes = ['awake', 'drowsy', 'sleeping']
        self.input_size = (64, 64)
        
        # User profile manager for personalization
        self.user_profile_manager = user_profile_manager
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        
        # NEW: Temporal components
        from .blink_tracker import BlinkTracker
        from .microsleep_detector import MicrosleepDetector
        from .temporal_features import TemporalFeatureExtractor
        
        self.blink_tracker = BlinkTracker()
        self.microsleep_detector = MicrosleepDetector()
        self.temporal_extractor = TemporalFeatureExtractor(window_size=30)
        
        # Frame buffer for LSTM (if used)
        self.frame_buffer = deque(maxlen=30)
        
        # LSTM model for temporal analysis
        self.lstm_model = None
        self.lstm_model_path = 'models/fatigue_lstm.keras'
        self._use_lstm = False
        self._init_lstm()
        
        # Load or build model
        self._init_model()
    
    def _init_model(self):
        """Initialize TensorFlow model."""
        if os.path.exists(self.model_path):
            self._load_model(self.model_path)
        else:
            self._build_cnn_model()
    
    def _init_lstm(self):
        """Initialize LSTM model for temporal analysis."""
        if os.path.exists(self.lstm_model_path):
            try:
                import tensorflow as tf
                self.lstm_model = tf.keras.models.load_model(self.lstm_model_path, compile=False)
                self._use_lstm = True
                print(f"LSTM Fatigue model loaded from {self.lstm_model_path}")
            except Exception as e:
                print(f"Failed to load LSTM model: {e}")
                self._use_lstm = False
        else:
            print("LSTM model not found, using CNN only")
            self._use_lstm = False
    def _load_model(self, model_path: str):
        """Load TensorFlow/Keras model."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"TF Fatigue model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load fatigue model: {e}")
            self._build_cnn_model()
    
    def _build_cnn_model(self):
        """Build CNN model for fatigue classification."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            model = models.Sequential([
                # Input
                layers.Input(shape=(64, 64, 1)),
                
                # Conv blocks
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Classification head
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            print("TF Fatigue CNN model built and ready")
            
        except Exception as e:
            print(f"Failed to build CNN model: {e}")
            self.model = None
    
    @property
    def is_ready(self) -> bool:
        return self.model is not None
    
    def _extract_eye_region(self, frame, face_landmarks):
        """
        Extract eye region from face using MediaPipe landmarks.
        
        Args:
            frame: BGR image from OpenCV
            face_landmarks: MediaPipe face landmarks (list or object with .landmark)
        
        Returns:
            grayscale image of eye region, or None
        """
        if face_landmarks is None:
            return None
        
        try:
            import cv2
            
            h, w = frame.shape[:2]
            
            # Handle both list and object formats
            if hasattr(face_landmarks, 'landmark'):
                # Object format (has .landmark attribute)
                landmarks = face_landmarks.landmark
            elif isinstance(face_landmarks, list):
                # List format
                landmarks = face_landmarks
            else:
                return None
            
            if len(landmarks) < 374:
                return None
            
            # Eye landmark indices
            left_eye_idx = [33, 133, 160, 144, 158, 153]
            right_eye_idx = [362, 263, 385, 380, 387, 373]
            all_eye_idx = left_eye_idx + right_eye_idx
            
            # Get bounding box - use .x and .y attributes
            x_coords = []
            y_coords = []
            for i in all_eye_idx:
                if i < len(landmarks):
                    x_coords.append(landmarks[i].x * w)
                    y_coords.append(landmarks[i].y * h)
            
            if not x_coords:
                return None
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            pad_x = int((x_max - x_min) * 0.5)
            pad_y = int((y_max - y_min) * 0.5)
            
            x_min = max(0, x_min - pad_x)
            x_max = min(w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h, y_max + pad_y)
            
            # Extract and convert to grayscale
            eye_region = frame[y_min:y_max, x_min:x_max]
            
            if eye_region.size == 0:
                return None
            
            eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            return eye_gray
            
        except Exception as e:
            return None
    
    def _preprocess_image(self, eye_region):
        """Preprocess image for CNN input."""
        if eye_region is None:
            return None
        
        try:
            import cv2
            
            # Resize
            img = cv2.resize(eye_region, self.input_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add dimensions
            img = np.expand_dims(img, axis=-1)  # Channel
            img = np.expand_dims(img, axis=0)    # Batch
            
            return img
            
        except Exception as e:
            return None
    
    def predict(self, face_landmarks, frame=None):
        """
        Predict fatigue level using TensorFlow CNN + temporal features.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame: BGR image from OpenCV
        
        Returns:
            dict with all required fields for UI:
            - status: 'awake', 'drowsy', 'sleeping'
            - confidence: prediction confidence
            - raw_scores: array of class probabilities
            - ear: Eye Aspect Ratio
            - mar: Mouth Aspect Ratio  
            - blink_rate: blinks per minute (real tracking)
            - fatigue_score: 0-100 score
            - fatigue_level: 'normal', 'mild', 'moderate', 'severe'
            - trend: 'stable', 'increasing', 'decreasing'
            - microsleep_detected: boolean
            - yawning: boolean
            - temporal_features: dict with temporal analysis
        """
        current_time = time.time()
        
        if face_landmarks is None:
            result = self._unknown_result()
            result['ear'] = 0.35
            result['mar'] = 0.0
            result['blink_rate'] = 0
            result['fatigue_score'] = 0
            result['fatigue_level'] = 'normal'
            result['trend'] = 'stable'
            result['microsleep_detected'] = False
            result['yawning'] = False
            return result
        
        # Calculate EAR and MAR
        ear = self._calculate_ear(face_landmarks)
        mar = self._calculate_mar(face_landmarks)
        
        # Get head pose features
        head_features = self._calculate_head_features(face_landmarks)
        
        # NEW: Update temporal components
        self.blink_tracker.update(ear, current_time)
        self.microsleep_detector.update(ear, current_time)
        
        temporal_features = self.temporal_extractor.update(
            ear=ear,
            mar=mar,
            head_droop=head_features.get('head_droop', 0),
            head_tilt=head_features.get('head_tilt', 0),
            current_time=current_time
        )
        
        # Get real blink rate from tracker
        blink_rate = self.blink_tracker.get_blink_rate_per_minute()
        
        # Get microsleep info
        microsleep_info = self.microsleep_detector.get_statistics()
        
        # Check yawning (from temporal features)
        yawning = temporal_features.get('yawning', False)
        
        # Extract eye region for CNN
        eye_region = self._extract_eye_region(frame, face_landmarks)
        
        # If we have valid eye region, run CNN prediction
        if eye_region is not None:
            # Try LSTM prediction first (uses temporal features)
            lstm_result = self._predict_lstm()
            
            # Add to LSTM buffer for next prediction
            buffer_ready = self._add_to_lstm_buffer(ear, mar, head_features, temporal_features)
            
            img = self._preprocess_image(eye_region)
            if img is not None:
                cnn_result = self._predict_cnn(img)
                cnn_result['ear'] = ear
                cnn_result['mar'] = mar
                cnn_result['blink_rate'] = blink_rate
                
                # Use LSTM result if available (better temporal analysis)
                if lstm_result is not None:
                    cnn_result['status'] = lstm_result['status']
                    cnn_result['confidence'] = lstm_result['confidence']
                    cnn_result['model_used'] = 'lstm'
                else:
                    cnn_result['model_used'] = 'cnn'
                
                # Add new temporal features
                cnn_result['microsleep_detected'] = microsleep_info['danger_level'] == 'danger'
                cnn_result['microsleep_count'] = microsleep_info['microsleeps_per_minute']
                cnn_result['yawning'] = yawning
                cnn_result['head_droop'] = head_features.get('head_droop', 0)
                cnn_result['head_tilt'] = head_features.get('head_tilt', 0)
                cnn_result['temporal_features'] = temporal_features
                
                # Convert status to fatigue_score (0-100)
                fatigue_score = self._status_to_fatigue_score(cnn_result['status'], cnn_result['confidence'])
                
                # Apply personalization if available
                if self.user_profile_manager:
                    personalization = self.user_profile_manager.apply_personalization(
                        ear, mar, blink_rate
                    )
                    fatigue_score = int(fatigue_score * personalization.get('personal_fatigue_factor', 1.0))
                
                cnn_result['fatigue_score'] = min(100, fatigue_score)
                cnn_result['fatigue_level'] = cnn_result['status']
                
                # Determine trend from temporal features
                if temporal_features.get('ear_trend', 0) < -0.01:
                    cnn_result['trend'] = 'decreasing'
                elif temporal_features.get('ear_trend', 0) > 0.01:
                    cnn_result['trend'] = 'increasing'
                else:
                    cnn_result['trend'] = 'stable'
                
                return cnn_result
        
        # Fallback to geometric calculation
        result = self._predict_geometric(face_landmarks, ear, mar)
        result['ear'] = ear
        result['mar'] = mar
        result['blink_rate'] = blink_rate
        result['fatigue_score'] = self._status_to_fatigue_score(result['status'], result['confidence'])
        result['fatigue_level'] = result['status']
        result['trend'] = 'stable'
        
        return result
    
    def _calculate_ear(self, face_landmarks):
        """Calculate Eye Aspect Ratio from face landmarks."""
        try:
            LEFT_EYE = [33, 133, 160, 144, 158, 153]
            RIGHT_EYE = [362, 263, 385, 380, 387, 373]
            
            # Handle both list and object formats
            if hasattr(face_landmarks, 'landmark'):
                landmarks = face_landmarks.landmark
            elif isinstance(face_landmarks, list):
                landmarks = face_landmarks
            else:
                return 0.35
            
            def get_point(idx):
                if hasattr(landmarks[idx], 'x'):
                    return np.array([landmarks[idx].x, landmarks[idx].y])
                else:
                    return np.array([landmarks[idx][0], landmarks[idx][1]])
            
            def eye_aspect_ratio(indices):
                p1 = get_point(indices[0])
                p4 = get_point(indices[1])
                p2 = get_point(indices[2])
                p6 = get_point(indices[3])
                p3 = get_point(indices[4])
                p5 = get_point(indices[5])
                
                v1 = np.linalg.norm(p2 - p6)
                v2 = np.linalg.norm(p3 - p5)
                h = np.linalg.norm(p1 - p4)
                
                return (v1 + v2) / (2.0 * h + 1e-6)
            
            left_ear = eye_aspect_ratio(LEFT_EYE)
            right_ear = eye_aspect_ratio(RIGHT_EYE)
            
            return (left_ear + right_ear) / 2.0
        except:
            return 0.35
    
    def _calculate_mar(self, face_landmarks):
        """Calculate Mouth Aspect Ratio from face landmarks."""
        try:
            MOUTH = [61, 291, 13, 14]
            
            # Handle both list and object formats
            if hasattr(face_landmarks, 'landmark'):
                landmarks = face_landmarks.landmark
            elif isinstance(face_landmarks, list):
                landmarks = face_landmarks
            else:
                return 0.15
            
            def get_point(idx):
                if hasattr(landmarks[idx], 'x'):
                    return np.array([landmarks[idx].x, landmarks[idx].y])
                else:
                    return np.array([landmarks[idx][0], landmarks[idx][1]])
            
            p_left = get_point(MOUTH[0])
            p_right = get_point(MOUTH[1])
            p_top = get_point(MOUTH[2])
            p_bottom = get_point(MOUTH[3])
            
            v = np.linalg.norm(p_top - p_bottom)
            h = np.linalg.norm(p_left - p_right)
            
            return v / (h + 1e-6)
        except:
            return 0.15
    
    def _calculate_blink_rate(self):
        """Get real blink rate from BlinkTracker."""
        if hasattr(self, 'blink_tracker'):
            return self.blink_tracker.get_blink_rate_per_minute()
        return 15
    
    def _calculate_head_features(self, face_landmarks):
        """Extract head pose features for fatigue detection."""
        try:
            # Handle both list and object formats
            if hasattr(face_landmarks, 'landmark'):
                landmarks = face_landmarks.landmark
            elif isinstance(face_landmarks, list):
                landmarks = face_landmarks
            else:
                return {'head_droop': 0, 'head_tilt': 0, 'is_drooping': False}
            
            if len(landmarks) < 455:
                return {'head_droop': 0, 'head_tilt': 0, 'is_drooping': False}
            
            # Key points for head pose
            nose = landmarks[1]
            chin = landmarks[152]
            forehead = landmarks[10]
            left_ear = landmarks[234]
            right_ear = landmarks[454]
            
            # Head droop (forward tilt) - normalized position of nose relative to face height
            face_height = chin.y - forehead.y
            if face_height > 0:
                nose_position = (nose.y - forehead.y) / face_height
                # Normal nose position is around 0.35-0.4, lower means droop
                head_droop = max(0, 0.4 - nose_position) * 90  # degrees
            
            # Head tilt (side tilt) - asymmetry of ears
            head_tilt = abs(right_ear.x - left_ear.x) * 45  # degrees
            
            return {
                'head_droop': float(head_droop),
                'head_tilt': float(head_tilt),
                'is_drooping': head_droop > 20
            }
            
        except Exception as e:
            return {'head_droop': 0, 'head_tilt': 0, 'is_drooping': False}
    
    def _add_to_lstm_buffer(self, ear, mar, head_features, temporal_features):
        """Add current frame features to LSTM buffer."""
        frame_data = {
            'ear_mean': ear,
            'ear_std': temporal_features.get('ear_std', 0.0),
            'ear_min': temporal_features.get('ear_min', ear),
            'ear_max': temporal_features.get('ear_max', ear),
            'ear_trend': temporal_features.get('ear_trend', 0.0),
            'mar_mean': mar,
            'mar_max': temporal_features.get('mar_max', mar),
            'mar_trend': temporal_features.get('mar_trend', 0.0),
            'eyes_closed_ratio': 1.0 if ear < 0.22 else 0.0,
            'eyes_very_closed_ratio': 1.0 if ear < 0.18 else 0.0,
            'yawning': float(temporal_features.get('yawning', False)),
            'yawn_intensity': temporal_features.get('yawn_intensity', 0.0),
            'ear_variance_long': temporal_features.get('ear_variance_long', 0.0),
            'ear_stability': temporal_features.get('ear_stability', 1.0),
            'estimated_blink_rate': temporal_features.get('estimated_blink_rate', 0) / 60.0,
            'head_droop': head_features.get('head_droop', 0) / 90.0,
        }
        self.frame_buffer.append(frame_data)
        
        return len(self.frame_buffer) >= 30
    
    def _predict_lstm(self):
        """Predict using LSTM model if available and buffer is full."""
        if not self._use_lstm or self.lstm_model is None:
            return None
            
        if len(self.frame_buffer) < 30:
            return None
        
        try:
            features = []
            for frame in self.frame_buffer:
                feature_vec = [
                    frame.get('ear_mean', 0.3),
                    frame.get('ear_std', 0.0),
                    frame.get('ear_min', 0.3),
                    frame.get('ear_max', 0.3),
                    frame.get('ear_trend', 0.0),
                    frame.get('mar_mean', 0.15),
                    frame.get('mar_max', 0.15),
                    frame.get('mar_trend', 0.0),
                    frame.get('eyes_closed_ratio', 0.0),
                    frame.get('eyes_very_closed_ratio', 0.0),
                    frame.get('yawning', 0.0),
                    frame.get('yawn_intensity', 0.0),
                    frame.get('ear_variance_long', 0.0),
                    frame.get('ear_stability', 1.0),
                    frame.get('estimated_blink_rate', 0.0),
                    frame.get('head_droop', 0.0),
                ]
                features.append(feature_vec)
            
            features = np.array(features, dtype=np.float32)
            features = np.expand_dims(features, axis=0)
            
            predictions = self.lstm_model.predict(features, verbose=0)[0]
            
            status_idx = np.argmax(predictions)
            status = self.classes[status_idx]
            confidence = float(predictions[status_idx])
            
            return {
                'status': status,
                'confidence': confidence,
                'probabilities': predictions.tolist(),
                'model_type': 'lstm'
            }
            
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return None
    
    def _status_to_fatigue_score(self, status, confidence):
        """Convert status to fatigue score 0-100."""
        status_scores = {
            'awake': 10,
            'drowsy': 50,
            'sleeping': 90,
            'unknown': 0
        }
        base = status_scores.get(status, 0)
        return min(100, base + (1 - confidence) * 20)
    
    def _predict_geometric(self, face_landmarks, ear, mar):
        """Fallback: use geometric features for classification."""
        # Geometric thresholds
        ear_threshold_low = 0.22
        ear_threshold_high = 0.28
        mar_threshold = 0.5
        
        if ear < ear_threshold_low:
            status = 'sleeping'
            confidence = 0.9
        elif ear < ear_threshold_high:
            if mar > mar_threshold:
                status = 'drowsy'
                confidence = 0.7
            else:
                status = 'drowsy'
                confidence = 0.5
        else:
            status = 'awake'
            confidence = 0.8
        
        # Smooth predictions
        self.prediction_history.append(status)
        smoothed_status = self._smooth_prediction()
        
        return {
            'status': smoothed_status,
            'confidence': confidence,
            'raw_scores': [0.0, 0.0, 0.0]
        }
    
    def _predict_cnn(self, img):
        """Run CNN prediction."""
        if self.model is None:
            return self._unknown_result()
        
        try:
            import tensorflow as tf
            
            # Run inference
            preds = self.model.predict(img, verbose=0)[0]
            
            # Get result
            status_idx = np.argmax(preds)
            status = self.classes[status_idx]
            confidence = float(preds[status_idx])
            
            # Smooth predictions
            self.prediction_history.append(status)
            smoothed_status = self._smooth_prediction()
            
            return {
                'status': smoothed_status,
                'confidence': confidence,
                'raw_scores': preds.tolist()
            }
            
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return self._unknown_result()
    
    def _smooth_prediction(self):
        """Apply smoothing to predictions using voting."""
        if len(self.prediction_history) < 3:
            return self.prediction_history[-1] if self.prediction_history else 'awake'
        
        from collections import Counter
        counts = Counter(self.prediction_history)
        return counts.most_common(1)[0][0]
    
    def _unknown_result(self):
        return {
            'status': 'unknown',
            'confidence': 0.0,
            'raw_scores': [0.0, 0.0, 0.0],
            'ear': 0.35,
            'mar': 0.0,
            'blink_rate': 0,
            'fatigue_score': 0,
            'fatigue_level': 'normal',
            'trend': 'stable'
        }
    
    def train(self, X_train, y_train, epochs: int = 50, save_path: str = None):
        """
        Train the classifier on user data.
        
        Args:
            X_train: images of shape (N, 64, 64)
            y_train: labels (N,) with values 0, 1, 2
            epochs: training epochs
            save_path: path to save trained model
        """
        if self.model is None:
            print("No model to train")
            return False
        
        try:
            import tensorflow as tf
            from tensorflow.keras.utils import to_categorical
            
            # Reshape if needed
            if len(X_train.shape) == 3:
                X_train = np.expand_dims(X_train, axis=-1)
            
            # Convert labels to categorical
            y_cat = to_categorical(y_train, num_classes=3)
            
            # Train
            history = self.model.fit(
                X_train, y_cat,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Save
            save_path = save_path or self.model_path
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def save_model(self, path: str = None):
        """Save the current model."""
        if self.model is None:
            print("No model to save")
            return False
        
        path = path or self.model_path
        try:
            self.model.save(path)
            print(f"Model saved to {path}")
            return True
        except Exception as e:
            print(f"Failed to save model: {e}")
            return False
    
    def get_status_text(self, status: str) -> str:
        """Get human-readable status text."""
        status_map = {
            'awake': 'Бодр',
            'drowsy': 'Сонлив',
            'sleeping': 'Засыпает',
            'unknown': 'Анализ...'
        }
        return status_map.get(status, 'Анализ...')


# Alias for backward compatibility
TFFatigueClassifier = FatigueClassifier