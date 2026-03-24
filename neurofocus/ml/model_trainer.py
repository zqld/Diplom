"""
Model Trainer for NeuroFocus ML classifiers.
Trains fatigue and posture models on collected user data.
"""

import os
import numpy as np
from neurofocus.ml import TrainingDataCollector
from neurofocus.ml.fatigue_classifier import FatigueClassifier
from neurofocus.ml.posture_classifier import PostureClassifier


class ModelTrainer:
    """
    Trains ML models on user-collected data.
    Supports incremental training and model updates.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.training_collector = TrainingDataCollector()
        os.makedirs(models_dir, exist_ok=True)
    
    def train_fatigue_model(self, epochs: int = 50, min_samples: int = 50) -> bool:
        """
        Train fatigue classifier on collected data.
        
        Args:
            epochs: Number of training epochs
            min_samples: Minimum samples required for training
        
        Returns:
            True if training successful
        """
        X, y = self.training_collector.get_training_data('fatigue')
        
        if X is None or len(X) < min_samples:
            print(f"Not enough fatigue samples: {len(X) if X is not None else 0}/{min_samples}")
            return False
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Fatigue training data: {dict(zip(unique, counts))}")
        
        # Need at least 2 samples per class ideally
        if len(counts) < 2 or min(counts) < 5:
            print("Warning: Unbalanced classes, training may be less accurate")
        
        # Create and train classifier
        classifier = FatigueClassifier()
        success = classifier.train(X, y, epochs=epochs, save_path=None)
        
        if success:
            # Save the trained model
            save_path = os.path.join(self.models_dir, 'fatigue_model.keras')
            try:
                classifier.model.save(save_path)
                print(f"Fatigue model saved to {save_path}")
            except Exception as e:
                print(f"Failed to save fatigue model: {e}")
        
        return success
    
    def train_posture_model(self, epochs: int = 50, min_samples: int = 50) -> bool:
        """
        Train posture classifier on collected data.
        
        Args:
            epochs: Number of training epochs
            min_samples: Minimum samples required for training
        
        Returns:
            True if training successful
        """
        X, y = self.training_collector.get_training_data('posture')
        
        if X is None or len(X) < min_samples:
            print(f"Not enough posture samples: {len(X) if X is not None else 0}/{min_samples}")
            return False
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Posture training data: {dict(zip(unique, counts))}")
        
        if len(counts) < 2 or min(counts) < 5:
            print("Warning: Unbalanced classes, training may be less accurate")
        
        # Create and train classifier
        classifier = PostureClassifier()
        success = classifier.train(X, y, epochs=epochs, save_path=None)
        
        if success:
            # Save the trained model
            save_path = os.path.join(self.models_dir, 'posture_model.keras')
            try:
                classifier.model.save(save_path)
                print(f"Posture model saved to {save_path}")
            except Exception as e:
                print(f"Failed to save posture model: {e}")
        
        return success
    
    def train_all(self, epochs: int = 50, min_samples: int = 50) -> dict:
        """
        Train all models.
        
        Returns:
            dict with training results for each model
        """
        results = {
            'fatigue': self.train_fatigue_model(epochs, min_samples),
            'posture': self.train_posture_model(epochs, min_samples)
        }
        
        return results
    
    def get_training_stats(self) -> dict:
        """Get statistics about available training data."""
        return self.training_collector.get_stats()
    
    def export_data(self, sample_type: str = None):
        """
        Export training data as CSV for external analysis.
        
        Args:
            sample_type: 'fatigue', 'posture', or None for both
        """
        if sample_type is None or sample_type == 'fatigue':
            self.training_collector.export_csv('fatigue')
        if sample_type is None or sample_type == 'posture':
            self.training_collector.export_csv('posture')
    
    def clear_training_data(self, sample_type: str = None):
        """
        Clear collected training data.
        
        Args:
            sample_type: 'fatigue', 'posture', or None for all
        """
        self.training_collector.clear_data(sample_type)
    
    def suggest_training(self) -> dict:
        """
        Check if training is recommended based on data.
        
        Returns:
            dict with suggestions
        """
        stats = self.training_collector.get_stats()
        
        suggestions = {
            'fatigue': {
                'ready': stats['fatigue']['total'] >= 50,
                'count': stats['fatigue']['total'],
                'suggestion': ''
            },
            'posture': {
                'ready': stats['posture']['total'] >= 50,
                'count': stats['posture']['total'],
                'suggestion': ''
            }
        }
        
        if suggestions['fatigue']['count'] < 50:
            suggestions['fatigue']['suggestion'] = f"Нужно ещё {50 - suggestions['fatigue']['count']} образцов"
        else:
            suggestions['fatigue']['suggestion'] = "Готов к обучению!"
        
        if suggestions['posture']['count'] < 50:
            suggestions['posture']['suggestion'] = f"Нужно ещё {50 - suggestions['posture']['count']} образцов"
        else:
            suggestions['posture']['suggestion'] = "Готов к обучению!"
        
        return suggestions


def main():
    """Command-line interface for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NeuroFocus ML models')
    parser.add_argument('--stats', action='store_true', help='Show training data statistics')
    parser.add_argument('--export', action='store_true', help='Export training data as CSV')
    parser.add_argument('--train', choices=['fatigue', 'posture', 'all'], default='all',
                        help='Which model to train')
    parser.add_argument('--clear', choices=['fatigue', 'posture', 'all'], default=None,
                        help='Clear training data')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--min-samples', type=int, default=50, help='Minimum samples required')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    if args.stats:
        stats = trainer.get_training_stats()
        print("Training Data Statistics:")
        print(f"  Fatigue samples: {stats['fatigue']['total']}")
        for label, count in stats['fatigue']['by_label'].items():
            print(f"    - {label}: {count}")
        print(f"  Posture samples: {stats['posture']['total']}")
        for label, count in stats['posture']['by_label'].items():
            print(f"    - {label}: {count}")
        print(f"  Corrections: {stats['corrections']['total']}")
        
        suggestions = trainer.suggest_training()
        print("\nTraining Suggestions:")
        print(f"  Fatigue: {suggestions['fatigue']['suggestion']}")
        print(f"  Posture: {suggestions['posture']['suggestion']}")
    
    elif args.export:
        trainer.export_data()
    
    elif args.clear:
        confirm = input(f"Clear {args.clear} training data? (y/n): ")
        if confirm.lower() == 'y':
            trainer.clear_training_data(args.clear)
            print("Data cleared")
    
    else:
        results = trainer.train_all(epochs=args.epochs, min_samples=args.min_samples)
        print("\nTraining Results:")
        for model, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {model}: {status}")


if __name__ == '__main__':
    main()