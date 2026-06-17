from PyQt6.QtWidgets import QApplication


def window_geometry(fraction=0.85, widget=None):
    """
    Возвращает (x, y, w, h) для центрированного окна.
    Использует availableGeometry() — исключает панель задач.
    Если передан widget — гарантирует размер >= minimumSize() виджета.
    """
    screen = QApplication.primaryScreen().availableGeometry()
    w = int(screen.width() * fraction)
    h = int(screen.height() * fraction)

    if widget is not None:
        min_sz = widget.minimumSize()
        w = max(w, min_sz.width(), 400)
        h = max(h, min_sz.height(), 300)

    w = min(w, screen.width())
    h = min(h, screen.height())
    x = screen.x() + (screen.width() - w) // 2
    y = screen.y() + (screen.height() - h) // 2
    return x, y, w, h
