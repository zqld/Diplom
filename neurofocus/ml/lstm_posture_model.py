"""
LSTM Posture Model - Temporal analysis for posture detection.
"""

import numpy as np
import os


class LSTMPostureModel:
    """
    LSTM-based posture classifier using temporal features.
    
    Classes: good, fair, bad
    """
    
    def __init__(self, model_path: str = 'models/posture_lstm.keras',
                 sequence_length: int = 30, num_features: int = 12):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.classes = ['good', 'fair', 'bad']
        
        self.model = None
        self._use_lstm = False
        self.frame_buffer = []
        
        self._load_or_build_model()
    
    def _load_or_build_model(self):
        """Load existing model or prepare for training."""
        if os.path.exists(self.model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                self._use_lstm = True
                print(f"LSTM Posture model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load posture LSTM: {e}")
                self._use_lstm = False
        else:
            print("Posture LSTM model not found, will use PostureClassifier")
            self._use_lstm = False
    
    @property
    def is_ready(self) -> bool:
        return self._use_lstm and self.model is not None
    
    def extract_features(self, posture_data: dict, temporal_data: dict = None) -> list:
        """
        Extract features for LSTM from posture data.
        """
        features = [
            posture_data.get('shoulder_angle', 0) / 45.0,
            posture_data.get('shoulder_diff', 0) / 20.0,
            posture_data.get('forward_lean', 0) / 30.0,
            posture_data.get('torso_tilt', 0) / 30.0,
            posture_data.get('head_tilt', 0) / 45.0,
            posture_data.get('head_droop', 0) / 45.0,
            posture_data.get('confidence', 0.8),
            1.0 if posture_data.get('status') == 'bad' else 0.0,
            1.0 if posture_data.get('status') == 'fair' else 0.0,
            1.0 if posture_data.get('status') == 'good' else 0.0,
            posture_data.get('score', 80) / 100.0,
            0.0,
        ]
        
        if temporal_data:
            features.extend([
                temporal_data.get('posture_stability', 1.0),
                temporal_data.get('posture_trend', 0.0),
            ])
        
        return features[:self.num_features]
    
    def update_buffer(self, features: list):
        """Add features to sequence buffer."""
        self.frame_buffer.append(features)
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)
    
    def predict(self) -> dict:
        """
        Predict posture from buffered sequence.
        """
        if not self._use_lstm or len(self.frame_buffer) < self.sequence_length:
            return {'status': 'unknown', 'confidence': 0.0, 'model': 'lstm_buffer_empty'}
        
        try:
            features = np.array(self.frame_buffer[-self.sequence_length:], dtype=np.float32)
            features = np.expand_dims(features, axis=0)
            
            predictions = self.model.predict(features, verbose=0)[0]
            
            status_idx = np.argmax(predictions)
            status = self.classes[status_idx]
            confidence = float(predictions[status_idx])
            
            return {
                'status': status,
                'confidence': confidence,
                'probabilities': predictions.tolist(),
                'model': 'lstm'
            }
            
        except Exception as e:
            print(f"Posture LSTM error: {e}")
            return {'status': 'unknown', 'confidence': 0.0, 'model': 'lstm_error'}
    
    def reset_buffer(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
