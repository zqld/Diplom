"""
Training Data Collector for ML classifiers.
Collects and stores labeled data for training fatigue and posture models.
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from collections import deque


class TrainingDataCollector:
    """
    Collects training data for fatigue and posture classifiers.
    Data is collected during normal app usage and can be used for retraining.
    """
    
    def __init__(self, data_dir: str = "data/ml_training"):
        self.data_dir = data_dir
        self._ensure_dir()
        
        # Fatigue data storage
        self.fatigue_samples = deque(maxlen=1000)
        
        # Posture data storage
        self.posture_samples = deque(maxlen=1000)
        
        # History of predictions for auto-labeling
        self.fatigue_history = deque(maxlen=50)
        self.posture_history = deque(maxlen=50)
        
        # Manual corrections (user feedback)
        self.corrections = deque(maxlen=100)
        
        # Auto-save interval
        self._last_save_time = time.time()
        self._save_interval = 60  # seconds
        
        # Load existing data
        self._load_existing_data()
    
    def _ensure_dir(self):
        """Ensure data directory exists."""
        fatigue_dir = os.path.join(self.data_dir, "fatigue")
        posture_dir = os.path.join(self.data_dir, "posture")
        
        os.makedirs(fatigue_dir, exist_ok=True)
        os.makedirs(posture_dir, exist_ok=True)
    
    def _load_existing_data(self):
        """Load existing training data from disk."""
        fatigue_file = os.path.join(self.data_dir, "fatigue", "samples.json")
        posture_file = os.path.join(self.data_dir, "posture", "samples.json")
        
        if os.path.exists(fatigue_file):
            try:
                with open(fatigue_file, 'r') as f:
                    data = json.load(f)
                    self.fatigue_samples = deque(data, maxlen=1000)
            except Exception as e:
                print(f"Failed to load fatigue data: {e}")
        
        if os.path.exists(posture_file):
            try:
                with open(posture_file, 'r') as f:
                    data = json.load(f)
                    self.posture_samples = deque(data, maxlen=1000)
            except Exception as e:
                print(f"Failed to load posture data: {e}")
    
    def add_fatigue_sample(self, features: dict, predicted_label: str, confidence: float):
        """
        Add a fatigue training sample.
        
        Args:
            features: dict with 'ear', 'mar', 'eye_region', etc.
            predicted_label: 'awake', 'drowsy', 'sleeping'
            confidence: prediction confidence
        """
        sample = {
            'timestamp': datetime.now().isoformat(),
            'label': predicted_label,
            'confidence': confidence,
            'features': {
                'ear': features.get('ear', 0.3),
                'mar': features.get('mar', 0.15),
                'blink_rate': features.get('blink_rate', 0),
                'eye_openness': features.get('eye_openness', 0.5),
            }
        }
        
        self.fatigue_samples.append(sample)
        self.fatigue_history.append({
            'label': predicted_label,
            'confidence': confidence,
            'time': time.time()
        })
        
        # Auto-save periodically
        self._check_auto_save()
    
    def add_posture_sample(self, features: dict, predicted_label: str, confidence: float):
        """
        Add a posture training sample.
        
        Args:
            features: dict with 'shoulder_angle', 'shoulder_diff', etc.
            predicted_label: 'good', 'fair', 'bad'
            confidence: prediction confidence
        """
        sample = {
            'timestamp': datetime.now().isoformat(),
            'label': predicted_label,
            'confidence': confidence,
            'features': {
                'shoulder_angle': features.get('shoulder_angle', 0),
                'shoulder_diff': features.get('shoulder_diff', 0),
                'forward_lean': features.get('forward_lean', 0),
                'torso_tilt': features.get('torso_tilt', 0),
            }
        }
        
        self.posture_samples.append(sample)
        self.posture_history.append({
            'label': predicted_label,
            'confidence': confidence,
            'time': time.time()
        })
        
        # Auto-save periodically
        self._check_auto_save()
    
    def add_correction(self, sample_type: str, features: dict, 
                      wrong_label: str, correct_label: str):
        """
        Add a manual correction (user feedback).
        
        Args:
            sample_type: 'fatigue' or 'posture'
            features: feature dict
            wrong_label: incorrectly predicted label
            correct_label: correct label (user feedback)
        """
        correction = {
            'timestamp': datetime.now().isoformat(),
            'type': sample_type,
            'features': features,
            'wrong_label': wrong_label,
            'correct_label': correct_label
        }
        
        self.corrections.append(correction)
    
    def get_training_data(self, sample_type: str):
        """
        Get training data as numpy arrays for model training.
        
        Args:
            sample_type: 'fatigue' or 'posture'
        
        Returns:
            X (features), y (labels) as numpy arrays
        """
        if sample_type == 'fatigue':
            samples = list(self.fatigue_samples)
            label_map = {'awake': 0, 'drowsy': 1, 'sleeping': 2}
        elif sample_type == 'posture':
            samples = list(self.posture_samples)
            label_map = {'good': 0, 'fair': 1, 'bad': 2}
        else:
            return None, None
        
        if not samples:
            return None, None
        
        # Extract features and labels
        X = []
        y = []
        
        for sample in samples:
            features = list(sample['features'].values())
            label = label_map.get(sample['label'], 0)
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def get_corrections_for_training(self, sample_type: str):
        """Get corrections as training data."""
        corrections = [c for c in self.corrections if c['type'] == sample_type]
        
        if not corrections:
            return None, None
        
        X = []
        y = []
        
        if sample_type == 'fatigue':
            label_map = {'awake': 0, 'drowsy': 1, 'sleeping': 2}
        else:
            label_map = {'good': 0, 'fair': 1, 'bad': 2}
        
        for corr in corrections:
            features = list(corr['features'].values())
            label = label_map.get(corr['correct_label'], 0)
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def get_stats(self) -> dict:
        """Get statistics about collected data."""
        fatigue_labels = {}
        posture_labels = {}
        
        for sample in self.fatigue_samples:
            label = sample['label']
            fatigue_labels[label] = fatigue_labels.get(label, 0) + 1
        
        for sample in self.posture_samples:
            label = sample['label']
            posture_labels[label] = posture_labels.get(label, 0) + 1
        
        return {
            'fatigue': {
                'total': len(self.fatigue_samples),
                'by_label': fatigue_labels
            },
            'posture': {
                'total': len(self.posture_samples),
                'by_label': posture_labels
            },
            'corrections': {
                'total': len(self.corrections)
            }
        }
    
    def _check_auto_save(self):
        """Auto-save data periodically."""
        if time.time() - self._last_save_time > self._save_interval:
            self.save()
            self._last_save_time = time.time()
    
    def save(self):
        """Save all collected data to disk."""
        # Save fatigue data
        fatigue_file = os.path.join(self.data_dir, "fatigue", "samples.json")
        with open(fatigue_file, 'w') as f:
            json.dump(list(self.fatigue_samples), f, indent=2)
        
        # Save posture data
        posture_file = os.path.join(self.data_dir, "posture", "samples.json")
        with open(posture_file, 'w') as f:
            json.dump(list(self.posture_samples), f, indent=2)
        
        # Save corrections
        corrections_file = os.path.join(self.data_dir, "corrections.json")
        with open(corrections_file, 'w') as f:
            json.dump(list(self.corrections), f, indent=2)
        
        print(f"Training data saved: {len(self.fatigue_samples)} fatigue, "
              f"{len(self.posture_samples)} posture samples")
    
    def clear_data(self, sample_type: str = None):
        """
        Clear collected data.
        
        Args:
            sample_type: 'fatigue', 'posture', or None for all
        """
        if sample_type is None or sample_type == 'fatigue':
            self.fatigue_samples.clear()
            fatigue_file = os.path.join(self.data_dir, "fatigue", "samples.json")
            if os.path.exists(fatigue_file):
                os.remove(fatigue_file)
        
        if sample_type is None or sample_type == 'posture':
            self.posture_samples.clear()
            posture_file = os.path.join(self.data_dir, "posture", "samples.json")
            if os.path.exists(posture_file):
                os.remove(posture_file)
        
        if sample_type is None:
            self.corrections.clear()
            corrections_file = os.path.join(self.data_dir, "corrections.json")
            if os.path.exists(corrections_file):
                os.remove(corrections_file)
        
        print(f"Data cleared for: {sample_type or 'all'}")
    
    def export_csv(self, sample_type: str, filename: str = None):
        """
        Export training data as CSV.
        
        Args:
            sample_type: 'fatigue' or 'posture'
            filename: output filename (auto-generated if None)
        """
        import csv
        
        if sample_type == 'fatigue':
            samples = list(self.fatigue_samples)
            feature_names = ['ear', 'mar', 'blink_rate', 'eye_openness']
            label_names = ['awake', 'drowsy', 'sleeping']
        elif sample_type == 'posture':
            samples = list(self.posture_samples)
            feature_names = ['shoulder_angle', 'shoulder_diff', 'forward_lean', 'torso_tilt']
            label_names = ['good', 'fair', 'bad']
        else:
            return
        
        if not samples:
            print(f"No {sample_type} samples to export")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{sample_type}_training_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = feature_names + ['label', 'confidence', 'timestamp']
            writer.writerow(header)
            
            # Data
            for sample in samples:
                features = list(sample['features'].values())
                row = features + [sample['label'], sample['confidence'], sample['timestamp']]
                writer.writerow(row)
        
        print(f"Exported {len(samples)} samples to {filepath}")