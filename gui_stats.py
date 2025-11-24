#создает окно, загружает данные и строит два графика с аналитикой по усталости и осанке
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt
from src.analytics import AnalyticsEngine, MplCanvas
import pandas as pd

STYLESHEET_STATS = """
QDialog { background-color: #1e1e1e; }
QLabel { color: white; font-family: 'Segoe UI'; font-size: 14px; }
QPushButton { 
    background-color: #007acc; color: white; 
    border-radius: 5px; padding: 8px; font-weight: bold;
}
QPushButton:hover { background-color: #005c99; }
"""

class StatsWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Аналитический отчет")
        self.resize(1000, 700)
        self.setStyleSheet(STYLESHEET_STATS)
        
        self.layout = QVBoxLayout(self)
        
        # Заголовок
        header = QLabel("Динамика сессии пользователя")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(header)

        # Место для графиков
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout.addWidget(self.canvas)
        
        # Статистика текстом
        self.stats_label = QLabel("Загрузка данных...")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #aaaaaa; font-size: 12px; margin: 10px;")
        self.layout.addWidget(self.stats_label)

        # Кнопка закрытия
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        self.layout.addWidget(btn_close)

        # Запуск построения
        self.plot_data()

    def plot_data(self):
        engine = AnalyticsEngine()
        df = engine.load_data()

        if df is None or len(df) < 10:
            self.stats_label.setText("Недостаточно данных для построения графиков (нужно минимум 10 секунд записи).")
            return

        # Очищаем оси
        ax1, ax2 = self.canvas.axes
        ax1.clear()
        ax2.clear()

        # --- ГРАФИК 1: ОСАНКА (Angle Pitch) ---
        # Сглаживаем данные (скользящее среднее), чтобы график не был "дерганым"
        # window=10 означает среднее за 10 секунд
        df['pitch_smooth'] = df['pitch'].rolling(window=5).mean()
        
        ax1.plot(df['timestamp'], df['pitch_smooth'], color='#00ff00', label='Наклон головы')
        # Рисуем линию порога (например, 10 градусов)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Порог плохой осанки')
        ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5)
        
        ax1.set_title('Анализ осанки (Угол наклона)', color='white')
        ax1.set_ylabel('Градусы', color='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.legend()
        ax1.grid(True, alpha=0.2)

        # --- ГРАФИК 2: УСТАЛОСТЬ (EAR + Зевки) ---
        df['ear_smooth'] = df['ear'].rolling(window=10).mean()
        
        ax2.plot(df['timestamp'], df['ear_smooth'], color='#00aaff', label='Открытость глаз (EAR)')
        
        # Находим моменты зевков для отображения точками
        yawns = df[df['fatigue_status'] == 'Yawning']
        if not yawns.empty:
            ax2.scatter(yawns['timestamp'], yawns['ear'], color='red', s=30, label='Зевки', zorder=5)

        ax2.set_title('Анализ усталости (Глаза и Зевки)', color='white')
        ax2.set_ylabel('EAR (0.0 - 0.4)', color='white')
        ax2.tick_params(axis='x', colors='white', labelrotation=45) # Поворот дат
        ax2.tick_params(axis='y', colors='white')
        ax2.legend()
        ax2.grid(True, alpha=0.2)

        # Обновляем холст
        self.canvas.draw()

        # --- ТЕКСТОВАЯ СТАТИСТИКА ---
        total_time = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).seconds
        yawn_count = len(yawns)
        # Считаем время плохой осанки (количество записей * 1 сек)
        bad_posture_sec = len(df[df['posture_status'] == 'Bad Posture'])
        
        stat_text = (
            f"📊 <b>ОБЩИЙ ОТЧЕТ:</b><br>"
            f"⏱ Общее время сессии: {total_time // 60} мин {total_time % 60} сек.<br>"
            f"🥱 Количество зевков: {yawn_count}<br>"
            f"🦐 Время с плохой осанкой: {bad_posture_sec} сек ({(bad_posture_sec/len(df)*100):.1f}% времени)."
        )
        self.stats_label.setText(stat_text)