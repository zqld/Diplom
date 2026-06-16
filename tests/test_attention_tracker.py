import time
import pytest
from src.attention_tracker import AttentionTracker


class TestAttentionTracker:
    def test_cold_start_returns_raw(self):
        t = AttentionTracker(window_seconds=600)
        assert t.update(85, 1000.0) == 85.0

    def test_single_sample(self):
        t = AttentionTracker(window_seconds=600)
        t.update(80, 1000.0)
        assert t.get_smoothed(1001.0) == 80.0

    def test_averaging(self):
        t = AttentionTracker(window_seconds=600, sample_interval=0.5)
        t.update(100, 1000.0)
        t.update(50, 1001.0)
        t.update(0, 1002.0)
        smoothed = t.get_smoothed(1002.5)
        # (100 + 50 + 0) / 3 = 50
        assert smoothed == 50.0

    def test_window_eviction(self):
        t = AttentionTracker(window_seconds=10, sample_interval=0.5)
        t.update(100, 1000.0)
        t.update(50, 1006.0)  # 6 sec later — stays in 10-sec window
        t.update(0, 1015.0)   # 15 sec later
        smoothed = t.get_smoothed(1016.0)
        # cutoff = 1006, so (1000, 100) pruned, (1006, 50) and (1015, 0) remain
        assert smoothed == 25.0  # (50 + 0) / 2 = 25

    def test_empty_after_window_returns_raw(self):
        t = AttentionTracker(window_seconds=10, sample_interval=1.0)
        t.update(80, 1000.0)
        # After 20 sec, everything is pruned
        assert t.get_smoothed(1020.0) == 80.0

    def test_sample_interval_respected(self):
        t = AttentionTracker(window_seconds=600, sample_interval=5.0)
        t.update(100, 1000.0)
        t.update(50, 1001.0)  # < 5 sec, should NOT be stored
        assert len(t._samples) == 1
        assert t._samples[0][1] == 100.0

    def test_reset(self):
        t = AttentionTracker(window_seconds=600)
        t.update(30, 1000.0)
        t.reset()
        assert len(t._samples) == 0
        assert t._last_raw == 100.0
        assert t.update(90, 1001.0) == 90.0

    def test_update_returns_smoothed(self):
        t = AttentionTracker(window_seconds=600, sample_interval=0.5)
        t.update(100, 1000.0)
        result = t.update(60, 1001.0)
        assert result == 80.0  # (100 + 60) / 2

    def test_fatigue_drop_does_not_instantly_crash_smoothed(self):
        """Закрыл глаза на 1 сек — smoothed почти не меняется."""
        t = AttentionTracker(window_seconds=600, sample_interval=0.2)
        # 10 samples at 100
        for i in range(10):
            t.update(100, 1000.0 + i * 0.2)
        # 5 samples at 5 (eyes closed, 1 sec)
        for i in range(5):
            t.update(5, 1002.0 + i * 0.2)
        smoothed = t.get_smoothed(1003.0)
        # 15 samples total: (10*100 + 5*5) / 15 ≈ 68.3
        assert smoothed > 60  # should still be relatively high
