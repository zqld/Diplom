"""
LSTM Fatigue Model - Temporal sequence analysis for fatigue detection.
"""

import numpy as np
import os


class LSTMFatigueModel:
    """
    LSTM-based fatigue classifier using temporal features.
    
    This model analyzes sequences of frames to detect fatigue patterns
    that single-frame classifiers might miss.
    """
    
    def __init__(self, model_path: str = 'models/fatigue_lstm.keras',
                 sequence_length: int = 30, 
                 num_features: int = 16):
        """
        Args:
            model_path: Path to saved LSTM model
            sequence_length: Number of frames in sequence (30 = 1 sec at 30fps)
            num_features: Number of features per frame
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.classes = ['awake', 'drowsy', 'sleeping']
        
        self.model = None
        self._model_loaded = False
        
        # Fallback to CNN if LSTM not available
        self._use_fallback = True
        self._fallback_model = None
        
        # Load model
        self._load_or_build_model()
    
    def _load_or_build_model(self):
        """Load existing model or build new one."""
        # First try to load LSTM model
        if os.path.exists(self.model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                self._model_loaded = True
                self._use_fallback = False
                print(f"LSTM Fatigue model loaded from {self.model_path}")
                return
            except Exception as e:
                print(f"Failed to load LSTM model: {e}")
        
        # Build model for training
        self._build_model()
        self._model_loaded = False
    
    def _build_model(self):
        """Build LSTM architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            model = models.Sequential([
                # Input: (sequence_length, num_features)
                layers.Input(shape=(self.sequence_length, self.num_features)),
                
                # First LSTM layer
                layers.LSTM(64, return_sequences=True, dropout=0.3),
                
                # Second LSTM layer
                layers.LSTM(32, return_sequences=False, dropout=0.3),
                
                # Dense layers
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.2),
                
                # Output: 3 classes
                layers.Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self._model_loaded = True
            print("LSTM Fatigue model built (ready for training)")
            
        except Exception as e:
            print(f"Failed to build LSTM model: {e}")
            self.model = None
    
    @property
    def is_ready(self) -> bool:
        """Check if model is ready for prediction."""
        return self.model is not None
    
    def predict(self, frame_sequence: list) -> dict:
        """
        Predict fatigue from sequence of frames.
        
        Args:
            frame_sequence: List of dicts with temporal features
            
        Returns:
            dict with prediction results
        """
        if self.model is None:
            return self._unknown_result()
        
        # Extract features from sequence
        features = self._extract_sequence_features(frame_sequence)
        
        # Reshape for LSTM: (1, sequence_length, num_features)
        features = np.expand_dims(features, axis=0)
        
        try:
            predictions = self.model.predict(features, verbose=0)[0]
            
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
            return self._unknown_result()
    
    def _extract_sequence_features(self, frame_sequence: list) -> np.ndarray:
        """
        Extract feature array from frame sequence.
        
        Args:
            frame_sequence: List of feature dicts
            
        Returns:
            numpy array of shape (sequence_length, num_features)
        """
        features = []
        
        for frame in frame_sequence:
            # Extract features from dict
            feature_vector = [
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
                float(frame.get('yawning', False)),
                frame.get('yawn_intensity', 0.0),
                frame.get('ear_variance_long', 0.0),
                frame.get('ear_stability', 1.0),
                frame.get('estimated_blink_rate', 0) / 60.0,
                0.0  # Padding
            ]
            
            features.append(feature_vector)
        
        # Pad sequence to sequence_length
        while len(features) < self.sequence_length:
            features.append([0.0] * self.num_features)
        
        # Trim to sequence_length
        features = features[:self.sequence_length]
        
        return np.array(features, dtype=np.float32)
    
    def _unknown_result(self) -> dict:
        """Return unknown result."""
        return {
            'status': 'unknown',
            'confidence': 0.0,
            'probabilities': [0.0, 0.0, 0.0],
            'model_type': 'lstm'
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 50, validation_split: float = 0.2,
              save_path: str = None):
        """
        Train LSTM model on sequence data.
        
        Args:
            X_train: Array of shape (samples, sequence_length, features)
            y_train: Labels (0, 1, 2) for awake, drowsy, sleeping
            epochs: Training epochs
            validation_split: Validation split ratio
            save_path: Path to save trained model
        """
        if self.model is None:
            print("No model to train")
            return False
        
        try:
            import tensorflow as tf
            from tensorflow.keras.utils import to_categorical
            
            # Convert labels to categorical
            y_cat = to_categorical(y_train, num_classes=3)
            
            # Train
            history = self.model.fit(
                X_train, y_cat,
                epochs=epochs,
                batch_size=32,
                validation_split=validation_split,
                verbose=1
            )
            
            # Save
            save_path = save_path or self.model_path
            self.model.save(save_path)
            print(f"LSTM model saved to {save_path}")
            
            return True
            
        except Exception as e:
            print(f"LSTM training failed: {e}")
            return False
    
    def save_model(self, path: str = None):
        """Save the current model."""
        if self.model is None:
            print("No model to save")
            return False
            
        path = path or self.model_path
        try:
            self.model.save(path)
            print(f"LSTM model saved to {path}")
            return True
        except Exception as e:
            print(f"Failed to save LSTM model: {e}")
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


# Alias
LSTMClassifier = LSTMFatigueModel