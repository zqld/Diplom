#создает окно, загружает данные и строит два графика с аналитикой по усталости и осанке
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QFrame, QDateTimeEdit, QSplitter, QWidget, QSizePolicy)
from PyQt6.QtCore import Qt, QDateTime
from src.analytics import AnalyticsEngine, MplCanvas
import pandas as pd
import matplotlib.dates as mdates


# --- CSS СТИЛИ ДЛЯ ОТЧЕТА ---
STYLESHEET_STATS = """
QDialog { background-color: #1e1e1e; }
QLabel { color: #cccccc; font-family: 'Segoe UI'; font-size: 14px; }
QLabel#Header { color: white; font-size: 18px; font-weight: bold; margin-bottom: 10px; }
QFrame#SidePanel { background-color: #252526; border-right: 1px solid #3d3d3d; }

/* Стили календарей и полей ввода */
QDateTimeEdit {
    background-color: #333333;
    color: white;
    border: 1px solid #444444;
    border-radius: 5px;
    padding: 5px;
    font-size: 13px;
}
QDateTimeEdit::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #444444;
    border-left-style: solid;
}

/* Кнопки */
QPushButton { 
    background-color: #007acc; color: white; 
    border-radius: 5px; padding: 8px; font-weight: bold;
    border: none;
}
QPushButton:hover { background-color: #005c99; }
QPushButton#Secondary { background-color: #3d3d3d; }
QPushButton#Secondary:hover { background-color: #4d4d4d; }
"""

class StatsWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFocus Analytics")
        self.resize(1200, 800)
        self.setStyleSheet(STYLESHEET_STATS)
        
        self.engine = AnalyticsEngine()
        
        # Основной Layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- ЛЕВАЯ ПАНЕЛЬ (НАСТРОЙКИ) ---
        side_panel = QFrame()
        side_panel.setObjectName("SidePanel")
        side_panel.setFixedWidth(300)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(20, 20, 20, 20)
        side_layout.setSpacing(15)

        # Заголовок настроек
        lbl_settings = QLabel("Период отчета")
        lbl_settings.setObjectName("Header")
        side_layout.addWidget(lbl_settings)

        # Выбор даты НАЧАЛА
        side_layout.addWidget(QLabel("С какого момента:"))
        self.dt_start = QDateTimeEdit(QDateTime.currentDateTime().addSecs(-3600)) # По умолчанию - час назад
        self.dt_start.setDisplayFormat("dd.MM.yyyy HH:mm:ss")
        self.dt_start.setCalendarPopup(True) # Выпадающий календарь
        side_layout.addWidget(self.dt_start)

        # Выбор даты КОНЦА
        side_layout.addWidget(QLabel("По какой момент:"))
        self.dt_end = QDateTimeEdit(QDateTime.currentDateTime())
        self.dt_end.setDisplayFormat("dd.MM.yyyy HH:mm:ss")
        self.dt_end.setCalendarPopup(True)
        side_layout.addWidget(self.dt_end)

        # Кнопка "Применить"
        self.btn_apply = QPushButton("Построить графики")
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.clicked.connect(self.plot_data)
        side_layout.addWidget(self.btn_apply)

        # Разделитель
        side_layout.addSpacing(20)
        
        # Блок статистики текстом
        lbl_stat_header = QLabel("Сводка:")
        lbl_stat_header.setStyleSheet("font-weight: bold; color: white;")
        side_layout.addWidget(lbl_stat_header)
        
        self.stats_label = QLabel("Выберите период и нажмите кнопку.")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #aaaaaa; font-size: 13px;")
        side_layout.addWidget(self.stats_label)

        # Растяжка вниз
        side_layout.addStretch()
        
        # Кнопка закрытия
        btn_close = QPushButton("Закрыть")
        btn_close.setObjectName("Secondary")
        btn_close.clicked.connect(self.close)
        side_layout.addWidget(btn_close)

        # Добавляем левую панель
        main_layout.addWidget(side_panel)

        # --- ПРАВАЯ ЧАСТЬ (ГРАФИКИ) ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Заголовок контента
        self.lbl_graph_header = QLabel("Динамика показателей")
        self.lbl_graph_header.setObjectName("Header")
        self.lbl_graph_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.lbl_graph_header)

        # Холст с графиками
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        content_layout.addWidget(self.canvas)

        main_layout.addWidget(content_widget)

        # Автозапуск построения (за последний час)
        self.plot_data()

    def plot_data(self):
        # Получаем даты из виджетов
        start_dt = self.dt_start.dateTime().toPyDateTime()
        end_dt = self.dt_end.dateTime().toPyDateTime()

        # Загружаем
        df = self.engine.load_data(start_dt, end_dt)

        if df is None or df.empty:
            self.stats_label.setText(f"Нет данных за выбранный период.\n\nПроверьте, что:\n1. Запись велась.\n2. Начальная дата меньше конечной.")
            # Очистка графиков
            self.canvas.axes[0].clear()
            self.canvas.axes[1].clear()
            self.canvas.draw()
            return

        # Очищаем оси
        ax1, ax2 = self.canvas.axes
        ax1.clear()
        ax2.clear()

        # Форматирование времени на оси X (часы:минуты)
        time_fmt = mdates.DateFormatter('%H:%M')

        # --- ГРАФИК 1: ОСАНКА ---
        # Сглаживание
        if len(df) > 10:
            df['pitch_smooth'] = df['pitch'].rolling(window=5, min_periods=1).mean()
        else:
            df['pitch_smooth'] = df['pitch']
        
        ax1.plot(df['timestamp'], df['pitch_smooth'], color='#00ff00', linewidth=2, label='Угол наклона')
        ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Порог (10°)')
        
        # Заливка зоны плохой осанки
        ax1.fill_between(df['timestamp'], df['pitch_smooth'], 10, where=(df['pitch_smooth'] < 10), 
                         color='red', alpha=0.2, interpolate=True)
        
        ax1.set_title('Осанка (Зеленая линия - наклон головы)', color='white', fontsize=10)
        ax1.set_ylabel('Градусы', color='white')
        ax1.grid(True, linestyle='--', alpha=0.2)
        ax1.xaxis.set_major_formatter(time_fmt)
        ax1.tick_params(colors='white')
        ax1.legend(facecolor='#2d2d2d', edgecolor='#2d2d2d', labelcolor='white')

        # --- ГРАФИК 2: УСТАЛОСТЬ ---
        if len(df) > 10:
            df['ear_smooth'] = df['ear'].rolling(window=10, min_periods=1).mean()
        else:
            df['ear_smooth'] = df['ear']
        
        ax2.plot(df['timestamp'], df['ear_smooth'], color='#00aaff', linewidth=2, label='EAR (Глаза)')
        
        # Точки зевков
        yawns = df[df['fatigue_status'] == 'Yawning']
        if not yawns.empty:
            ax2.scatter(yawns['timestamp'], yawns['ear'], color='red', s=40, marker='x', label='Зевок', zorder=5)

        ax2.set_title('Усталость (Синяя линия - открытость глаз)', color='white', fontsize=10)
        ax2.set_ylabel('EAR', color='white')
        ax2.grid(True, linestyle='--', alpha=0.2)
        ax2.xaxis.set_major_formatter(time_fmt)
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#2d2d2d', edgecolor='#2d2d2d', labelcolor='white')

        self.canvas.draw()

        # --- СТАТИСТИКА ---
        total_records = len(df)
        bad_posture_count = len(df[df['posture_status'] == 'Bad Posture'])
        posture_percent = (bad_posture_count / total_records * 100) if total_records > 0 else 0
        
        stat_text = (
            f"📅 <b>ВЫБРАННЫЙ ПЕРИОД:</b><br>"
            f"С: {start_dt.strftime('%H:%M:%S')}<br>"
            f"По: {end_dt.strftime('%H:%M:%S')}<br><br>"
            f"⏱ Записей найдено: {total_records}<br>"
            f"🥱 Зевков: {len(yawns)}<br>"
            f"🦐 Плохая осанка: {posture_percent:.1f}% времени"
        )
        self.stats_label.setText(stat_text)