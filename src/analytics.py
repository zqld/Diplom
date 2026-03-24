#создает движок аналитики для загрузки данных из БД и виджет для отображения графиков в PyQt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.spines import Spine
from sqlalchemy import create_engine, text
import os


class AnalyticsEngine:
    def __init__(self, db_path="session_data.db"):
        self.db_uri = f"sqlite:///data/{db_path}"
    
    def load_data(self, start_time=None, end_time=None):
        """
        Загружает данные в Pandas DataFrame с фильтрацией по времени.
        start_time, end_time: объекты datetime
        """
        try:
            engine = create_engine(self.db_uri)
            
            # Базовый запрос
            query = "SELECT * FROM face_logs"
            params = {}
            
            # Если переданы даты, добавляем условие WHERE
            if start_time and end_time:
                query += " WHERE timestamp BETWEEN :start AND :end"
                params = {"start": start_time, "end": end_time}
            
            query += " ORDER BY timestamp ASC"
            
            # Читаем SQL с параметрами (безопасно)
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            if df.empty:
                return None
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Ошибка чтения БД: {e}")
            return None


MODERN_COLORS = {
    'background': '#2A2A32',
    'text': '#E8E8EC',
    'text_secondary': '#9898A0',
    'grid': '#404048',
    'accent': '#6B7A8F',
    'accent_light': '#8A9AAD',
    'good': '#5DAB6D',
    'warning': '#E8A54B',
    'danger': '#D45D5D',
    'posture_line': '#7A8A9A',
    'ear_line': '#8A9AAA',
    'yawn_marker': '#D45D5D',
    'threshold': '#E88A5B',
    'fill_good': '#1E3A25',
    'fill_bad': '#3A1E1E',
}


def setup_modern_axes(ax, title="", ylabel="", show_legend=False):
    """Apply modern minimalist styling to axes."""
    ax.set_title(title, color=MODERN_COLORS['text'], fontsize=12, fontweight='600', pad=12, loc='left')
    ax.set_ylabel(ylabel, color=MODERN_COLORS['text_secondary'], fontsize=10, labelpad=8)
    
    ax.set_facecolor(MODERN_COLORS['background'])
    ax.grid(True, linestyle='-', alpha=0.3, color=MODERN_COLORS['grid'], zorder=0)
    
    ax.tick_params(colors=MODERN_COLORS['text_secondary'], labelsize=9)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(MODERN_COLORS['grid'])
    ax.spines['bottom'].set_linewidth(1)
    
    if show_legend:
        ax.legend(loc='upper right', frameon=True, facecolor=MODERN_COLORS['background'], 
                  edgecolor=MODERN_COLORS['grid'], labelcolor=MODERN_COLORS['text_secondary'],
                  fontsize=9, framealpha=1)


# Виджет для встраивания графика в PyQt
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig, self.axes = plt.subplots(2, 1, figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#2A2A32')
        
        for ax in self.axes:
            ax.set_facecolor('#2A2A32')
            
        self.fig.tight_layout(pad=3.0)
        super(MplCanvas, self).__init__(self.fig)
    
    def apply_modern_style(self):
        """Apply consistent modern styling across all axes."""
        for ax in self.axes:
            ax.set_facecolor('#2A2A32')
            ax.grid(True, linestyle='-', alpha=0.3, color=MODERN_COLORS['grid'], zorder=0)
            ax.tick_params(colors=MODERN_COLORS['text_secondary'], labelsize=9)
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_color(MODERN_COLORS['grid'])
            ax.spines['bottom'].set_linewidth(1)
    
    def draw_empty(self, message="Нет данных"):
        """Draw empty state with message."""
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor('#2A2A32')
            ax.text(0.5, 0.5, message, transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14, 
                   color=MODERN_COLORS['text_secondary'],
                   fontfamily='Segoe UI')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        self.draw()
