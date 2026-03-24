"""
Final Integration Test for NeuroFocus ML System.
Tests the complete ML pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fatigue_classifier():
    """Test FatigueClassifier (TensorFlow CNN)."""
    print("Testing FatigueClassifier...")
    
    from neurofocus.ml import FatigueClassifier
    
    fc = FatigueClassifier()
    assert fc.is_ready, "Model should be ready"
    assert len(fc.classes) == 3, "Should have 3 classes"
    assert fc.classes == ['awake', 'drowsy', 'sleeping']
    
    # Test with None input
    result = fc.predict(None)
    assert result['status'] == 'unknown'
    
    print("  [OK] FatigueClassifier working")
    return True


def test_posture_classifier():
    """Test PostureClassifier (TensorFlow Hub)."""
    print("Testing PostureClassifier...")
    
    from neurofocus.ml import PostureClassifier
    
    pc = PostureClassifier(use_tf_hub=True)
    assert pc._tf_hub_estimator is not None
    assert pc._tf_hub_estimator.is_available, "MoveNet should be available"
    assert len(pc.classes) == 3, "Should have 3 classes"
    assert pc.classes == ['good', 'fair', 'bad']
    
    # Test with None input
    result = pc.predict(None)
    assert result['status'] == 'unknown'
    
    print("  [OK] PostureClassifier with MoveNet working")
    return True


def test_training_data():
    """Test TrainingDataCollector."""
    print("Testing TrainingDataCollector...")
    
    from neurofocus.ml import TrainingDataCollector
    
    tdc = TrainingDataCollector()
    
    # Add sample
    tdc.add_fatigue_sample(
        features={'ear': 0.3, 'mar': 0.1},
        predicted_label='awake',
        confidence=0.9
    )
    
    stats = tdc.get_stats()
    assert stats['fatigue']['total'] >= 1
    
    print("  [OK] TrainingDataCollector working")
    return True


def test_main_window_integration():
    """Test MainWindow ML integration."""
    print("Testing MainWindow integration...")
    
    from neurofocus.windows.main_window import VideoThread
    
    vt = VideoThread()
    
    assert hasattr(vt, 'fatigue_classifier')
    assert hasattr(vt, 'posture_classifier')
    assert hasattr(vt, 'training_collector')
    
    assert vt.fatigue_classifier.is_ready
    assert vt.posture_classifier._tf_hub_estimator.is_available
    
    vt._run_flag = False
    
    print("  [OK] MainWindow ML integration working")
    return True


def test_tf_hub_pose():
    """Test TFHubPoseEstimator."""
    print("Testing TFHubPoseEstimator...")
    
    from neurofocus.ml import TFHubPoseEstimator
    
    pose = TFHubPoseEstimator()
    assert pose.is_available, "MoveNet should be available"
    
    print("  [OK] TFHubPoseEstimator working")
    return True


def main():
    print("=" * 60)
    print("NEUROFOCUS ML SYSTEM - FINAL INTEGRATION TEST")
    print("=" * 60)
    print()
    
    tests = [
        ("FatigueClassifier", test_fatigue_classifier),
        ("PostureClassifier", test_posture_classifier),
        ("TFHubPoseEstimator", test_tf_hub_pose),
        ("TrainingDataCollector", test_training_data),
        ("MainWindow", test_main_window_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, False))
        print()
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    passed = sum(1 for _, r in results if r)
    print()
    print(f"Total: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print()
        print("ALL SYSTEMS OPERATIONAL!")
        return True
    else:
        print()
        print("SOME TESTS FAILED!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
