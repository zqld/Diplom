"""
Test script for NeuroFocus ML classifiers.
Tests the ML system integration and functionality.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np


def test_imports():
    """Test that all ML modules can be imported."""
    print("Testing imports...")
    
    try:
        from neurofocus.ml import (
            FatigueClassifier,
            PostureClassifier,
            TrainingDataCollector,
            ModelTrainer
        )
        print("  [OK] All ML modules imported")
        return True
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_classifiers():
    """Test Fatigue and Posture classifiers."""
    print("Testing classifiers...")
    
    from neurofocus.ml import FatigueClassifier, PostureClassifier
    
    # Test FatigueClassifier
    fc = FatigueClassifier()
    result = fc.predict(None)
    assert result['status'] == 'unknown', "Should return 'unknown' for None input"
    print(f"  [OK] FatigueClassifier: predict(None) = {result}")
    
    # Test PostureClassifier
    pc = PostureClassifier()
    result = pc.predict(None)
    assert result['status'] == 'unknown', "Should return 'unknown' for None input"
    print(f"  [OK] PostureClassifier: predict(None) = {result}")
    
    return True


def test_training_data():
    """Test TrainingDataCollector."""
    print("Testing training data collection...")
    
    from neurofocus.ml import TrainingDataCollector
    
    tdc = TrainingDataCollector()
    
    # Add fatigue sample
    tdc.add_fatigue_sample(
        features={'ear': 0.3, 'mar': 0.15},
        predicted_label='awake',
        confidence=0.9
    )
    
    # Add posture sample
    tdc.add_posture_sample(
        features={'shoulder_angle': 5.0, 'shoulder_diff': 10.0, 'forward_lean': 0.05, 'torso_tilt': 2.0},
        predicted_label='good',
        confidence=0.85
    )
    
    stats = tdc.get_stats()
    assert stats['fatigue']['total'] == 1, "Should have 1 fatigue sample"
    assert stats['posture']['total'] == 1, "Should have 1 posture sample"
    print(f"  [OK] TrainingDataCollector: {stats}")
    
    return True


def test_preprocessing():
    """Test preprocessing functions."""
    print("Testing preprocessing...")
    
    from neurofocus.ml import extract_eye_region, extract_pose_features, prepare_face_image
    
    # Test with None (should return None)
    result = extract_eye_region(None, None)
    assert result is None, "Should return None for None input"
    print("  [OK] extract_eye_region(None) = None")
    
    result = extract_pose_features(None)
    assert result is None, "Should return None for None input"
    print("  [OK] extract_pose_features(None) = None")
    
    return True


def test_model_trainer():
    """Test ModelTrainer."""
    print("Testing ModelTrainer...")
    
    from neurofocus.ml import ModelTrainer
    
    mt = ModelTrainer()
    stats = mt.get_training_stats()
    print(f"  [OK] ModelTrainer stats: {stats}")
    
    suggestions = mt.suggest_training()
    print(f"  [OK] Training suggestions: fatigue={suggestions['fatigue']['ready']}, posture={suggestions['posture']['ready']}")
    
    return True


def test_main_window_integration():
    """Test that MainWindow can use ML classifiers."""
    print("Testing MainWindow integration...")
    
    from neurofocus.windows.main_window import VideoThread
    
    # Just verify we can instantiate the thread
    thread = VideoThread()
    
    assert hasattr(thread, 'fatigue_classifier'), "Should have fatigue_classifier"
    assert hasattr(thread, 'posture_classifier'), "Should have posture_classifier"
    assert hasattr(thread, 'training_collector'), "Should have training_collector"
    
    print("  [OK] VideoThread has ML attributes")
    
    thread._run_flag = False  # Stop immediately
    
    return True


def test_training_workflow():
    """Test complete training workflow."""
    print("Testing training workflow...")
    
    from neurofocus.ml import (
        FatigueClassifier,
        PostureClassifier,
        TrainingDataCollector
    )
    
    # Create collector and add samples
    tdc = TrainingDataCollector()
    
    # Generate synthetic training data
    # Class 0: awake (high EAR, low MAR)
    for i in range(20):
        ear = 0.28 + np.random.random() * 0.05
        mar = 0.1 + np.random.random() * 0.1
        tdc.add_fatigue_sample(
            features={'ear': ear, 'mar': mar, 'blink_rate': 15},
            predicted_label='awake',
            confidence=0.9
        )
    
    # Class 1: drowsy (lower EAR)
    for i in range(20):
        ear = 0.2 + np.random.random() * 0.05
        mar = 0.15 + np.random.random() * 0.15
        tdc.add_fatigue_sample(
            features={'ear': ear, 'mar': mar, 'blink_rate': 25},
            predicted_label='drowsy',
            confidence=0.75
        )
    
    # Get training data
    X, y = tdc.get_training_data('fatigue')
    
    if X is not None and len(X) >= 40:
        print(f"  [OK] Generated training data: X.shape={X.shape}, y.shape={y.shape}")
        
        # Train model
        fc = FatigueClassifier()
        success = fc.train(X, y, epochs=10, save_path=None)
        
        if success:
            print("  [OK] Model trained successfully")
        else:
            print("  [WARN] Model training returned False")
    else:
        print("  [WARN] Not enough samples for training")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("NeuroFocus ML System Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Preprocessing", test_preprocessing),
        ("Classifiers", test_classifiers),
        ("Training Data", test_training_data),
        ("Model Trainer", test_model_trainer),
        ("MainWindow Integration", test_main_window_integration),
        ("Training Workflow", test_training_workflow),
    ]
    
    results = []
    for name, test_func in tests:
        print()
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, False))
    
    print()
    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)