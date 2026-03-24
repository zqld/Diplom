import pytest
import numpy as np


class TestFatigueAnalyzer:
    def test_initialization(self):
        from neurofocus.analyzers.fatigue import FatigueAnalyzer
        analyzer = FatigueAnalyzer()
        assert analyzer is not None
        assert analyzer.window_size == 30
    
    def test_update_returns_dict(self):
        from neurofocus.analyzers.fatigue import FatigueAnalyzer
        analyzer = FatigueAnalyzer()
        result = analyzer.update(0.35, 0.15, 0.0, "Neutral", 1000.0)
        assert isinstance(result, dict)
        assert 'fatigue_score' in result
        assert 'fatigue_level' in result


class TestPostureAnalyzer:
    def test_initialization(self):
        from neurofocus.analyzers.posture import PostureAnalyzer
        analyzer = PostureAnalyzer()
        assert analyzer is not None
    
    def test_default_metrics(self):
        from neurofocus.analyzers.posture import PostureAnalyzer
        analyzer = PostureAnalyzer()
        result = analyzer.update_from_face_mesh(None)
        assert isinstance(result, dict)
        assert 'posture_score' in result
        assert result['posture_level'] == 'unknown'


class TestGeometry:
    def test_calculate_ear(self):
        from neurofocus.utils.geometry import calculate_ear, euclidean_distance, get_coords
        
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        landmarks = [MockLandmark(0.5, 0.5)] * 500
        
        landmarks_list = [
            (33, 0.3, 0.4), (133, 0.7, 0.4),
            (160, 0.35, 0.38), (144, 0.35, 0.42),
            (158, 0.65, 0.38), (153, 0.65, 0.42),
            (362, 0.7, 0.4), (263, 0.3, 0.4),
            (385, 0.65, 0.38), (380, 0.65, 0.42),
            (387, 0.35, 0.38), (373, 0.35, 0.42),
        ]
        for idx, x, y in landmarks_list:
            landmarks[idx] = MockLandmark(x, y)
        
        ear = calculate_ear(landmarks)
        assert isinstance(ear, (int, float))
        assert ear > 0


class TestSettingsManager:
    def test_singleton(self):
        from neurofocus.core.settings_manager import settings_manager
        from neurofocus.core.settings_manager import SettingsManager
        assert isinstance(settings_manager, SettingsManager)
    
    def test_settings_dict(self):
        from neurofocus.core.settings_manager import settings_manager
        settings = settings_manager.settings
        assert isinstance(settings, dict)
        assert 'theme' in settings
        assert 'sound_volume' in settings
    
    def test_get_set(self):
        from neurofocus.core.settings_manager import settings_manager
        old_value = settings_manager.get('test_key', 'default')
        settings_manager.set('test_key', 'test_value')
        assert settings_manager.get('test_key') == 'test_value'
        settings_manager.set('test_key', old_value)


class TestThemeManager:
    def test_singleton(self):
        from neurofocus.ui.theme import theme_manager
        from neurofocus.ui.theme import ThemeManager
        assert isinstance(theme_manager, ThemeManager)
    
    def test_themes_exist(self):
        from neurofocus.ui.theme import theme_manager
        assert 'dark' in theme_manager.themes
        assert 'light' in theme_manager.themes
    
    def test_toggle_theme(self):
        from neurofocus.ui.theme import theme_manager
        old_theme = theme_manager.current_theme
        new_theme = theme_manager.toggle_theme()
        assert new_theme != old_theme
        theme_manager.set_theme(old_theme)


class TestSoundManager:
    def test_initialization(self):
        from neurofocus.utils.sound import sound_manager
        assert sound_manager is not None
        assert hasattr(sound_manager, 'enabled')
        assert hasattr(sound_manager, 'volume')
    
    def test_volume_setting(self):
        from neurofocus.utils.sound import sound_manager
        old_volume = sound_manager.volume
        sound_manager.set_volume(0.8)
        assert sound_manager.volume == 0.8
        sound_manager.set_volume(old_volume)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
