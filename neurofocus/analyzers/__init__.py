"""
Analyzers - Emotion analysis.
Note: Fatigue and Posture analysis is handled by ML classifiers in neurofocus.ml
"""

from .emotion import EmotionDetector as EmotionAnalyzer

__all__ = ["EmotionAnalyzer"]
