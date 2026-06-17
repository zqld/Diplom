from PyQt6.QtWidgets import QApplication


def window_geometry(fraction=0.85, widget=None):
    """
    Возвращает (x, y, w, h) для центрированного окна.
    Использует availableGeometry() — исключает панель задач.
    Если передан widget — гарантирует размер >= minimumSize() виджета.
    """
    return window_geometry_positioned(fraction, widget, 'center')


def window_geometry_positioned(fraction=0.85, widget=None, align='center'):
    """
    Возвращает (x, y, w, h) для окна с выравниванием.
    align: 'center' | 'bottom-right' | 'bottom-left'
    """
    screen = QApplication.primaryScreen().availableGeometry()
    w = int(screen.width() * fraction)
    h = int(screen.height() * fraction)

    if widget is not None:
        min_sz = widget.minimumSize()
        w = max(w, min_sz.width(), 300)
        h = max(h, min_sz.height(), 200)

    w = min(w, screen.width())
    h = min(h, screen.height())

    if align == 'center':
        x = screen.x() + (screen.width() - w) // 2
        y = screen.y() + (screen.height() - h) // 2
    elif align == 'bottom-right':
        x = screen.x() + screen.width() - w - 20
        y = screen.y() + screen.height() - h - 20
    elif align == 'bottom-left':
        x = screen.x() + 20
        y = screen.y() + screen.height() - h - 20
    else:
        x = screen.x() + (screen.width() - w) // 2
        y = screen.y() + (screen.height() - h) // 2

    return x, y, w, h
