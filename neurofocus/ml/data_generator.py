"""
Synthetic Fatigue Data Generator.
Generates realistic training data for LSTM model based on physiological patterns.
"""

import numpy as np
import os
import json
from datetime import datetime


class FatigueDataGenerator:
    """
    Generates synthetic fatigue data for training LSTM models.
    Based on realistic physiological patterns.
    """
    
    def __init__(self, output_dir: str = "data/ml_training/lstm"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # State: 0 = awake, 1 = drowsy, 2 = sleeping
        self.state = 0
        self.state_duration = 0
        
    def generate_sequence(self, length: int = 30, target_state: int = None) -> tuple:
        """
        Generate a sequence of frames with specified fatigue state.
        
        Args:
            length: Sequence length (30 frames = 1 second at 30fps)
            target_state: 0=awake, 1=drowsy, 2=sleeping (random if None)
            
        Returns:
            (features_array, label) tuple
        """
        if target_state is None:
            target_state = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        
        features = []
        
        for i in range(length):
            frame_features = self._generate_frame_features(target_state, i)
            features.append(frame_features)
            
        return np.array(features), target_state
    
    def _generate_frame_features(self, state: int, frame_idx: int) -> list:
        """Generate features for a single frame based on state."""
        
        # Base EAR values by state (with noise)
        if state == 0:  # Awake
            base_ear = np.random.uniform(0.28, 0.40)
            ear_noise = np.random.normal(0, 0.02)
        elif state == 1:  # Drowsy
            base_ear = np.random.uniform(0.18, 0.28)
            ear_noise = np.random.normal(0, 0.03)
        else:  # Sleeping
            base_ear = np.random.uniform(0.10, 0.20)
            ear_noise = np.random.normal(0, 0.02)
        
        ear = max(0.05, base_ear + ear_noise)
        
        # Base MAR values (yawning more common when drowsy)
        if state == 2:  # Sleeping - mouth may be open
            base_mar = np.random.uniform(0.15, 0.45)
        elif state == 1:  # Drowsy - occasional yawning
            if np.random.random() < 0.1:
                base_mar = np.random.uniform(0.40, 0.60)
            else:
                base_mar = np.random.uniform(0.12, 0.25)
        else:  # Awake
            base_mar = np.random.uniform(0.08, 0.20)
        
        mar = max(0.05, base_mar + np.random.normal(0, 0.02))
        
        # Head droop (more when tired)
        if state == 2:
            head_droop = np.random.uniform(15, 35)
        elif state == 1:
            head_droop = np.random.uniform(5, 20)
        else:
            head_droop = np.random.uniform(0, 10)
        
        # Blink rate (higher when drowsy)
        if state == 2:
            blink_rate = np.random.uniform(0, 5)
        elif state == 1:
            blink_rate = np.random.uniform(15, 35)
        else:
            blink_rate = np.random.uniform(10, 25)
        
        # Feature vector (16 features)
        features = [
            ear,                                    # ear_mean
            abs(ear_noise) * 2,                     # ear_std
            ear * 0.95,                             # ear_min
            ear * 1.05,                             # ear_max
            -0.01 if state > 0 else 0.001,          # ear_trend
            mar,                                    # mar_mean
            max(0.3, mar),                          # mar_max
            -0.005 if state > 0 else 0.001,         # mar_trend
            0.0 if state == 0 else 0.2,             # eyes_closed_ratio
            0.0 if state < 2 else 0.1,              # eyes_very_closed_ratio
            1.0 if mar > 0.4 else 0.0,             # yawning
            mar if mar > 0.3 else 0.0,              # yawn_intensity
            0.01 if state > 0 else 0.005,           # ear_variance_long
            50 if state == 0 else 30,               # ear_stability
            blink_rate / 60.0,                      # normalized_blink_rate
            head_droop / 90.0,                      # normalized_head_droop
        ]
        
        return features
    
    def generate_dataset(self, num_samples: int = 1000, sequence_length: int = 30) -> tuple:
        """
        Generate a complete dataset.
        
        Args:
            num_samples: Number of sequences to generate
            sequence_length: Frames per sequence
            
        Returns:
            (X, y) tuple for training
        """
        print(f"Generating {num_samples} sequences...")
        
        X = []
        y = []
        
        for i in range(num_samples):
            # Balanced classes
            state = i % 3
            features, label = self.generate_sequence(sequence_length, state)
            X.append(features)
            y.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} sequences")
        
        return np.array(X), np.array(y)
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, name: str = "fatigue_lstm"):
        """Save dataset to files."""
        X_path = os.path.join(self.output_dir, f"{name}_X.npy")
        y_path = os.path.join(self.output_dir, f"{name}_y.npy")
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        # Save metadata
        meta = {
            'name': name,
            'created': datetime.now().isoformat(),
            'num_samples': len(X),
            'sequence_length': X.shape[1],
            'num_features': X.shape[2],
            'class_distribution': {
                'awake': int(np.sum(y == 0)),
                'drowsy': int(np.sum(y == 1)),
                'sleeping': int(np.sum(y == 2))
            }
        }
        
        meta_path = os.path.join(self.output_dir, f"{name}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Dataset saved to {self.output_dir}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: awake={meta['class_distribution']['awake']}, "
              f"drowsy={meta['class_distribution']['drowsy']}, "
              f"sleeping={meta['class_distribution']['sleeping']}")


def generate_and_save_dataset():
    """Generate and save a training dataset."""
    generator = FatigueDataGenerator()
    
    print("=" * 50)
    print("Fatigue LSTM Data Generator")
    print("=" * 50)
    
    # Generate dataset
    num_samples = int(input("Number of samples per class (default 500): ") or "500")
    sequence_length = int(input("Sequence length (frames, default 30): ") or "30")
    
    X, y = generator.generate_dataset(num_samples * 3, sequence_length)
    
    # Save
    generator.save_dataset(X, y, "fatigue_lstm")
    
    print("\nDataset generated successfully!")
    return X, y


if __name__ == "__main__":
    generate_and_save_dataset()
