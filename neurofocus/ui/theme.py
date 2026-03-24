from typing import Dict


class ThemeManager:
    """Manages application themes (light/dark)."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._current_theme = "dark"
        
        self.themes: Dict[str, Dict[str, str]] = {
            "dark": {
                'bg_main': '#121215',
                'bg_card': '#1C1C22',
                'bg_input': '#252530',
                'text_primary': '#E8E8EC',
                'text_secondary': '#9898A4',
                'text_muted': '#606068',
                'accent': '#5B7FFF',
                'accent_hover': '#4A6AEF',
                'border': '#2A2A32',
                'border_light': '#383842',
                'good': '#4ADE80',
                'warning': '#FBBF24',
                'danger': '#F87171',
                'good_bg': 'rgba(74, 222, 128, 0.15)',
                'warning_bg': 'rgba(251, 191, 36, 0.15)',
                'danger_bg': 'rgba(248, 113, 113, 0.15)',
            },
            "light": {
                'bg_main': '#ECECEC',
                'bg_card': '#FAFAFA',
                'bg_input': '#DEDEDE',
                'text_primary': '#2D2D2D',
                'text_secondary': '#5A5A5A',
                'text_muted': '#8A8A8A',
                'accent': '#5B7FFF',
                'accent_hover': '#4A6AEF',
                'border': '#CCCCCC',
                'border_light': '#DDDDDD',
                'good': '#2E8B57',
                'warning': '#CD853F',
                'danger': '#B22222',
                'good_bg': 'rgba(46, 139, 87, 0.15)',
                'warning_bg': 'rgba(205, 133, 63, 0.15)',
                'danger_bg': 'rgba(178, 34, 34, 0.15)',
            }
        }
    
    @property
    def current_theme(self) -> str:
        return self._current_theme
    
    @property
    def colors(self) -> Dict[str, str]:
        return self.themes[self._current_theme]
    
    def set_theme(self, theme: str):
        if theme in self.themes:
            self._current_theme = theme
    
    def get_stylesheet(self, widget_type: str = "dialog") -> str:
        """Get base stylesheet for a widget type."""
        c = self.colors
        
        if widget_type == "main":
            return f"""
                QMainWindow, QDialog, QWidget {{
                    background-color: {c['bg_main']};
                    font-family: 'Segoe UI', sans-serif;
                }}
                QLabel {{
                    color: {c['text_primary']};
                }}
                QScrollBar:vertical {{
                    background: {c['bg_card']};
                    width: 6px;
                    border-radius: 3px;
                }}
                QScrollBar::handle:vertical {{
                    background: {c['border_light']};
                    border-radius: 3px;
                    min-height: 20px;
                }}
            """
        elif widget_type == "button":
            return f"""
                QPushButton {{
                    background-color: {c['bg_card']};
                    color: {c['text_secondary']};
                    border: 1px solid {c['border']};
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {c['bg_input']};
                    border-color: {c['border_light']};
                }}
            """
        elif widget_type == "card":
            return f"""
                QFrame {{
                    background-color: {c['bg_card']};
                    border: 1px solid {c['border']};
                    border-radius: 12px;
                }}
            """
        
        return ""
    
    def get_button_style(self, primary: bool = False, danger: bool = False, custom_color: str = None) -> str:
        """Get button stylesheet."""
        c = self.colors
        
        if danger:
            return f"""
                QPushButton {{
                    background-color: transparent;
                    color: {c['danger']};
                    border: 1px solid {c['danger']};
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {c['danger_bg']};
                }}
            """
        elif custom_color:
            return f"""
                QPushButton {{
                    background-color: {custom_color};
                    color: #FFFFFF;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    opacity: 0.85;
                }}
            """
        elif primary:
            return f"""
                QPushButton {{
                    background-color: {c['accent']};
                    color: #FFFFFF;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {c['accent_hover']};
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: {c['bg_card']};
                    color: {c['text_secondary']};
                    border: 1px solid {c['border']};
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {c['bg_input']};
                    border-color: {c['border_light']};
                }}
            """
    
    def get_progressbar_style(self) -> str:
        """Get progress bar stylesheet."""
        c = self.colors
        return f"""
            QProgressBar {{
                border: none;
                background-color: {c['bg_input']};
                border-radius: 6px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {c['accent']};
                border-radius: 6px;
            }}
        """
    
    def get_menu_style(self) -> str:
        """Get menu stylesheet."""
        c = self.colors
        return f"""
            QMenu {{
                background-color: {c['bg_card']};
                color: {c['text_primary']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {c['bg_input']};
            }}
        """
    
    def get_checkbox_style(self) -> str:
        """Get checkbox stylesheet."""
        c = self.colors
        return f"""
            QCheckBox {{
                color: {c['text_primary']};
                spacing: 12px;
                font-size: 14px;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 2px solid {c['border']};
                background: {c['bg_input']};
            }}
            QCheckBox::indicator:checked {{
                background: {c['accent']};
                border-color: {c['accent']};
            }}
        """
    
    def get_slider_style(self) -> str:
        """Get slider stylesheet."""
        c = self.colors
        return f"""
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {c['border']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {c['accent']};
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {c['accent_hover']};
            }}
        """
    
    def get_complete_stylesheet(self) -> str:
        """Get complete application stylesheet."""
        c = self.colors
        return f"""
            QMainWindow, QDialog, QWidget {{
                background-color: {c['bg_main']};
                color: {c['text_primary']};
                font-family: 'Segoe UI', sans-serif;
            }}
            QLabel {{
                color: {c['text_primary']};
                background: transparent;
            }}
            QPushButton {{
                background-color: {c['bg_card']};
                color: {c['text_secondary']};
                border: 1px solid {c['border']};
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {c['bg_input']};
                border-color: {c['border_light']};
            }}
            QScrollBar:vertical {{
                background: {c['bg_card']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['border_light']};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {c['text_muted']};
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {c['border']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {c['accent']};
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {c['accent_hover']};
            }}
            QCheckBox {{
                color: {c['text_primary']};
                spacing: 12px;
                font-size: 14px;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 2px solid {c['border']};
                background: {c['bg_input']};
            }}
            QCheckBox::indicator:checked {{
                background: {c['accent']};
                border-color: {c['accent']};
            }}
            QProgressBar {{
                border: none;
                background-color: {c['bg_input']};
                border-radius: 6px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {c['accent']};
                border-radius: 6px;
            }}
            QMenu {{
                background-color: {c['bg_card']};
                color: {c['text_primary']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {c['bg_input']};
            }}
            QStatusBar {{
                background-color: {c['bg_card']};
                color: {c['text_secondary']};
            }}
        """


theme_manager = ThemeManager()
