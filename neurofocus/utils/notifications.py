import sys
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from PyQt6.QtGui import QColor, QPalette, QFont
from sqlalchemy import create_engine, text
import datetime

try:
    from neurofocus.ui.theme import theme_manager
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

DEFAULT_SETTINGS = {
    "work_limit_minutes": 45,
    "posture_window_minutes": 3,
    "posture_bad_percent": 30,
    "yawn_limit": 4,
    "yawn_window_minutes": 10,
    "minimize_to_tray": True,
}


def get_toast_colors():
    if THEME_AVAILABLE:
        c = theme_manager.colors
        is_dark = theme_manager.current_theme == 'dark'
        return {
            'background': c['bg_card'],
            'border': c['border'],
            'accent': c['accent'],
            'title_color': c['accent'],
            'text_color': c['text_secondary'],
        }
    else:
        return {
            'background': '#2d2d2d',
            'border': '#444444',
            'accent': '#007acc',
            'title_color': '#00aaff',
            'text_color': '#cccccc',
        }


class ToastNotification(QWidget):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                            Qt.WindowType.Tool | 
                            Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.setFixedSize(300, 100)
        self._apply_styles()
        
        layout = QVBoxLayout(self)
        
        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("Title")
        layout.addWidget(self.lbl_title)
        
        self.lbl_msg = QLabel(message)
        self.lbl_msg.setObjectName("Msg")
        self.lbl_msg.setWordWrap(True)
        layout.addWidget(self.lbl_msg)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        self.close_timer = QTimer(self)
        self.close_timer.setInterval(5000)
        self.close_timer.timeout.connect(self.fade_out)
    
    def _apply_styles(self):
        colors = get_toast_colors()
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {colors['background']};
                border: 1px solid {colors['border']};
                border-left: 5px solid {colors['accent']};
                border-radius: 5px;
            }}
            QLabel {{ color: {colors['text_color']}; border: none; }}
            QLabel#Title {{ font-weight: bold; font-size: 14px; color: {colors['title_color']}; }}
            QLabel#Msg {{ font-size: 12px; color: {colors['text_color']}; }}
        """)
    
    def refresh_theme(self):
        self._apply_styles()
    
    def show_toast(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 20
        self.move(x, y)
        
        self._apply_styles()
        self.show()
        self.anim.start()
        self.close_timer.start()

    def fade_out(self):
        self.anim.setDirection(QPropertyAnimation.Direction.Backward)
        self.anim.finished.connect(self.close)
        self.anim.start()


class NotificationManager:
    def __init__(self, db_path="session_data.db"):
        self.db_uri = f"sqlite:///data/{db_path}"
        self.engine = create_engine(self.db_uri)
        
        self.settings = DEFAULT_SETTINGS.copy()
        
        self.session_start = datetime.datetime.now()
        
        self.last_alert_time = {
            "posture": datetime.datetime.min,
            "fatigue": datetime.datetime.min,
            "break": datetime.datetime.min
        }

    def update_settings(self, new_settings):
        self.settings.update(new_settings)

    def check_conditions(self):
        now = datetime.datetime.now()
        conn = self.engine.connect()
        alert = None

        try:
            work_duration = (now - self.session_start).total_seconds() / 60
            if work_duration > self.settings["work_limit_minutes"]:
                if (now - self.last_alert_time["break"]).total_seconds() > 600:
                    self.last_alert_time["break"] = now
                    return "Пора отдохнуть", f"Вы работаете уже {int(work_duration)} минут без перерыва. Сделайте разминку."

            window_min = self.settings["posture_window_minutes"]
            time_threshold = now - datetime.timedelta(minutes=window_min)
            
            sql_posture = text("""
                SELECT posture_status FROM face_logs 
                WHERE timestamp > :thresh
            """)
            result = conn.execute(sql_posture, {"thresh": time_threshold}).fetchall()
            
            if result:
                total_records = len(result)
                bad_count = sum(1 for r in result if r[0] == 'Bad Posture')
                bad_percent = (bad_count / total_records) * 100
                
                if bad_percent > self.settings["posture_bad_percent"]:
                    if (now - self.last_alert_time["posture"]).total_seconds() > 300:
                        self.last_alert_time["posture"] = now
                        return "Следите за осанкой", f"За последние {window_min} мин вы сутулились {int(bad_percent)}% времени."

            window_yawn = self.settings["yawn_window_minutes"]
            time_threshold_yawn = now - datetime.timedelta(minutes=window_yawn)
            
            sql_yawn = text("""
                SELECT COUNT(*) FROM face_logs 
                WHERE timestamp > :thresh AND fatigue_status = 'Yawning'
            """)
            yawn_count = conn.execute(sql_yawn, {"thresh": time_threshold_yawn}).scalar()
            
            if yawn_count >= self.settings["yawn_limit"]:
                if (now - self.last_alert_time["fatigue"]).total_seconds() > 600:
                    self.last_alert_time["fatigue"] = now
                    return "Обнаружена усталость", f"Вы часто зеваете ({yawn_count} раз за {window_yawn} мин). Рекомендуется проветрить помещение."

        except Exception as e:
            print(f"Ошибка анализатора: {e}")
        finally:
            conn.close()
        
        return None
