"""
NeuroFocus Windows - GUI Windows.
"""

try:
    from .main_window import MainWindow
except ImportError:
    MainWindow = None

try:
    from .settings import SettingsWindow
except ImportError:
    SettingsWindow = None

try:
    from .stats import StatsWindow
except ImportError:
    StatsWindow = None

try:
    from .progress import ProgressWindow
except ImportError:
    ProgressWindow = None

try:
    from .pomodoro import PomodoroWindow
except ImportError:
    PomodoroWindow = None

try:
    from .help import GestureHelpWindow
except ImportError:
    GestureHelpWindow = None

__all__ = [
    "MainWindow",
    "SettingsWindow", 
    "StatsWindow",
    "ProgressWindow",
    "PomodoroWindow",
    "GestureHelpWindow",
]
