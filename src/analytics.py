#создает движок аналитики для загрузки данных из БД и виджет для отображения графиков в PyQt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sqlalchemy import create_engine
import os

class AnalyticsEngine:
    def __init__(self, db_path="session_data.db"):
        self.db_uri = f"sqlite:///data/{db_path}"
    
    def load_data(self):
        """Загружает данные в Pandas DataFrame"""
        try:
            engine = create_engine(self.db_uri)
            # Читаем SQL таблицу сразу в DataFrame
            df = pd.read_sql("SELECT * FROM face_logs", engine)
            
            if df.empty:
                return None
                
            # Делаем колонку времени индексом для удобной группировки
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Ошибка чтения БД: {e}")
            return None

# Виджет для встраивания графика в PyQt
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Создаем фигуру (Figure) и оси (Axes)
        # dark_background - чтобы сочеталось с твоим интерфейсом
        with plt.style.context("dark_background"):
            self.fig, self.axes = plt.subplots(2, 1, figsize=(width, height), dpi=dpi)
            super(MplCanvas, self).__init__(self.fig)
            self.fig.tight_layout(pad=3.0) # Отступы между графиками