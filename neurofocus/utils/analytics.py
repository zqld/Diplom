#создает движок аналитики для загрузки данных из БД и виджет для отображения графиков в PyQt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.spines import Spine
from sqlalchemy import create_engine, text
import os

try:
    from neurofocus.ui.theme import theme_manager
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False


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
            
            query = "SELECT * FROM face_logs"
            params = {}
            
            if start_time and end_time:
                query += " WHERE timestamp BETWEEN :start AND :end"
                params = {"start": start_time, "end": end_time}
            
            query += " ORDER BY timestamp ASC"
            
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            if df.empty:
                return None
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Ошибка чтения БД: {e}")
            return None


def get_chart_colors():
    if THEME_AVAILABLE:
        c = theme_manager.colors
        is_dark = theme_manager.current_theme == 'dark'
        return {
            'background': '#1E1E25' if is_dark else '#F0F0F0',
            'text': c['text_primary'],
            'text_secondary': c['text_secondary'],
            'grid': c['border'],
            'accent': c['accent'],
            'accent_light': c['accent_hover'],
            'good': c['good'],
            'warning': c['warning'],
            'danger': c['danger'],
            'posture_line': '#9CA3AF',
            'ear_line': '#8A8A9A',
            'yawn_marker': c['danger'],
            'threshold': c['warning'],
            'fill_good': 'rgba(74,222,128,0.2)',
            'fill_bad': 'rgba(248,113,113,0.2)',
        }
    else:
        return {
            'background': '#1E1E25',
            'text': '#FFFFFF',
            'text_secondary': '#A0A0B0',
            'grid': '#3A3A45',
            'accent': '#6B8AFE',
            'accent_light': '#8AA3FF',
            'good': '#4ADE80',
            'warning': '#FBBF24',
            'danger': '#F87171',
            'posture_line': '#9CA3AF',
            'ear_line': '#8A8A9A',
            'yawn_marker': '#F87171',
            'threshold': '#FBBF24',
            'fill_good': 'rgba(74,222,128,0.2)',
            'fill_bad': 'rgba(248,113,113,0.2)',
        }


def setup_modern_axes(ax, title="", ylabel="", show_legend=False, colors=None):
    if colors is None:
        colors = get_chart_colors()
    
    ax.set_title(title, color=colors['text'], fontsize=12, fontweight='600', pad=12, loc='left')
    ax.set_ylabel(ylabel, color=colors['text_secondary'], fontsize=10, labelpad=8)
    
    ax.set_facecolor(colors['background'])
    ax.grid(True, linestyle='-', alpha=0.3, color=colors['grid'], zorder=0)
    
    ax.tick_params(colors=colors['text_secondary'], labelsize=9)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(colors['grid'])
    ax.spines['bottom'].set_linewidth(1)
    
    if show_legend:
        ax.legend(loc='upper right', frameon=True, facecolor=colors['background'], 
                  edgecolor=colors['grid'], labelcolor=colors['text_secondary'],
                  fontsize=9, framealpha=1)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig, self.axes = plt.subplots(2, 1, figsize=(width, height), dpi=dpi)
        self._update_figure_colors()
        
        for ax in self.axes:
            self._update_axes_colors(ax)
            
        self.fig.tight_layout(pad=3.0)
        super(MplCanvas, self).__init__(self.fig)
    
    def _update_figure_colors(self):
        colors = get_chart_colors()
        self.fig.patch.set_facecolor(colors['background'])
    
    def _update_axes_colors(self, ax, colors=None):
        if colors is None:
            colors = get_chart_colors()
        ax.set_facecolor(colors['background'])
        ax.grid(True, linestyle='-', alpha=0.3, color=colors['grid'], zorder=0)
        ax.tick_params(colors=colors['text_secondary'], labelsize=9)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color(colors['grid'])
        ax.spines['bottom'].set_linewidth(1)
    
    def apply_modern_style(self):
        colors = get_chart_colors()
        for ax in self.axes:
            self._update_axes_colors(ax, colors)
    
    def refresh_theme(self):
        colors = get_chart_colors()
        self.fig.patch.set_facecolor(colors['background'])
        for ax in self.axes:
            self._update_axes_colors(ax, colors)
        self.draw()
    
    def draw_empty(self, message="Нет данных"):
        colors = get_chart_colors()
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(colors['background'])
            ax.text(0.5, 0.5, message, transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14, 
                   color=colors['text_secondary'],
                   fontfamily='Segoe UI')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        self.draw()
