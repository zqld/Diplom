import sys
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from PyQt6.QtGui import QColor, QPalette, QFont
from sqlalchemy import create_engine, text
import datetime

# --- КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ ---
# Эти настройки можно будет менять из GUI
DEFAULT_SETTINGS = {
    "work_limit_minutes": 45,       # Уведомление о перерыве каждые 45 мин
    "posture_window_minutes": 3,    # Анализировать последние 3 минуты
    "posture_bad_percent": 30,      # Если > 30% времени осанка плохая -> АЛЕРТ
    "yawn_limit": 4,                # Если > 4 зевков за 10 минут -> АЛЕРТ
    "yawn_window_minutes": 10       # Анализировать последние 10 минут на зевки
}

# --- 1. КРАСИВОЕ УВЕДОМЛЕНИЕ (TOAST) ---
class ToastNotification(QWidget):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                            Qt.WindowType.Tool | 
                            Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Стилизация
        self.setFixedSize(300, 100)
        self.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-left: 5px solid #007acc;
                border-radius: 5px;
            }
            QLabel { color: white; border: none; }
            QLabel#Title { font-weight: bold; font-size: 14px; color: #00aaff; }
            QLabel#Msg { font-size: 12px; color: #cccccc; }
        """)

        layout = QVBoxLayout(self)
        
        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("Title")
        layout.addWidget(self.lbl_title)
        
        self.lbl_msg = QLabel(message)
        self.lbl_msg.setObjectName("Msg")
        self.lbl_msg.setWordWrap(True)
        layout.addWidget(self.lbl_msg)

        # Анимация появления
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        # Таймер закрытия (через 5 секунд)
        self.close_timer = QTimer(self)
        self.close_timer.setInterval(5000)
        self.close_timer.timeout.connect(self.fade_out)

    def show_toast(self):
        # Позиционирование справа внизу
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 20
        self.move(x, y)
        
        self.show()
        self.anim.start()
        self.close_timer.start()

    def fade_out(self):
        self.anim.setDirection(QPropertyAnimation.Direction.Backward)
        self.anim.finished.connect(self.close)
        self.anim.start()

# --- 2. МЕНЕДЖЕР ЛОГИКИ ---
class NotificationManager:
    def __init__(self, db_path="session_data.db"):
        self.db_uri = f"sqlite:///data/{db_path}"
        self.engine = create_engine(self.db_uri)
        
        # Настройки
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Время начала работы (для таймера перерыва)
        self.session_start = datetime.datetime.now()
        
        # Кулдауны (чтобы не спамить уведомлениями каждые 5 секунд)
        self.last_alert_time = {
            "posture": datetime.datetime.min,
            "fatigue": datetime.datetime.min,
            "break": datetime.datetime.min
        }

    def update_settings(self, new_settings):
        """Обновление настроек из GUI"""
        self.settings.update(new_settings)

    def check_conditions(self):
        """
        Главный метод анализа. Вызывается раз в минуту (или чаще).
        Возвращает (Title, Message) или None.
        """
        now = datetime.datetime.now()
        conn = self.engine.connect()
        alert = None

        try:
            # 1. ПРОВЕРКА ВРЕМЕНИ РАБОТЫ (Time Limit)
            # Если прошло больше времени, чем указано в настройках
            work_duration = (now - self.session_start).total_seconds() / 60
            if work_duration > self.settings["work_limit_minutes"]:
                # Если с последнего уведомления прошло > 10 минут
                if (now - self.last_alert_time["break"]).total_seconds() > 600:
                    self.last_alert_time["break"] = now
                    # Сбрасываем таймер сессии, будто человек отдохнул (условно)
                    # Либо просто напоминаем
                    return "Пора отдохнуть", f"Вы работаете уже {int(work_duration)} минут без перерыва. Сделайте разминку."

            # 2. АНАЛИЗ ОСАНКИ (За последние N минут)
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
                
                # Если процент плохой осанки выше порога
                if bad_percent > self.settings["posture_bad_percent"]:
                    # Кулдаун 5 минут
                    if (now - self.last_alert_time["posture"]).total_seconds() > 300:
                        self.last_alert_time["posture"] = now
                        return "Следите за осанкой", f"За последние {window_min} мин вы сутулились {int(bad_percent)}% времени."

            # 3. АНАЛИЗ УСТАЛОСТИ (Зевки)
            window_yawn = self.settings["yawn_window_minutes"]
            time_threshold_yawn = now - datetime.timedelta(minutes=window_yawn)
            
            sql_yawn = text("""
                SELECT COUNT(*) FROM face_logs 
                WHERE timestamp > :thresh AND fatigue_status = 'Yawning'
            """)
            yawn_count = conn.execute(sql_yawn, {"thresh": time_threshold_yawn}).scalar()
            
            if yawn_count >= self.settings["yawn_limit"]:
                # Кулдаун 10 минут
                if (now - self.last_alert_time["fatigue"]).total_seconds() > 600:
                    self.last_alert_time["fatigue"] = now
                    return "Обнаружена усталость", f"Вы часто зеваете ({yawn_count} раз за {window_yawn} мин). Рекомендуется проветрить помещение."

        except Exception as e:
            print(f"Ошибка анализатора: {e}")
        finally:
            conn.close()
        
        return None