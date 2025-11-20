import numpy as np

# Индексы точек для MediaPipe FaceMesh (стандартные значения)
# Глаза (Left и Right зеркальны)
# Формат: [левый уголок, правый уголок, верхняя точка 1, нижняя точка 1, верхняя 2, нижняя 2]
LEFT_EYE_IDXS = [33, 133, 160, 144, 158, 153]
RIGHT_EYE_IDXS = [362, 263, 385, 380, 387, 373]

# Рот (для зевания)
# [левый уголок, правый уголок, верхняя губа, нижняя губа]
MOUTH_IDXS = [61, 291, 13, 14]

def get_coords(landmarks, idx):
    """Возвращает (x, y) конкретной точки по индексу."""
    # landmarks[idx].x и .y - это относительные координаты (0.0 - 1.0)
    return np.array([landmarks[idx].x, landmarks[idx].y])

def euclidean_distance(point1, point2):
    """Считает расстояние между двумя точками (теорема Пифагора)."""
    return np.linalg.norm(point1 - point2)

def calculate_ear(landmarks):
    """
    Считает EAR (Eye Aspect Ratio) для обоих глаз.
    Формула: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Где числитель - вертикальные линии века, знаменатель - горизонтальная линия глаза.
    """
    
    def eye_aspect_ratio(indices):
        # Собираем координаты всех нужных точек глаза
        p1 = get_coords(landmarks, indices[0]) # Левый уголок
        p4 = get_coords(landmarks, indices[1]) # Правый уголок
        
        p2 = get_coords(landmarks, indices[2]) # Верх 1
        p6 = get_coords(landmarks, indices[3]) # Низ 1
        
        p3 = get_coords(landmarks, indices[4]) # Верх 2
        p5 = get_coords(landmarks, indices[5]) # Низ 2

        # Считаем расстояния
        vertical_1 = euclidean_distance(p2, p6)
        vertical_2 = euclidean_distance(p3, p5)
        horizontal = euclidean_distance(p1, p4)

        # Вычисляем EAR
        # Добавляем 1e-6 в знаменатель, чтобы избежать деления на ноль
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
        return ear

    left_ear = eye_aspect_ratio(LEFT_EYE_IDXS)
    right_ear = eye_aspect_ratio(RIGHT_EYE_IDXS)

    # Возвращаем среднее по двум глазам
    return (left_ear + right_ear) / 2.0

def calculate_mar(landmarks):
    """
    Считает MAR (Mouth Aspect Ratio) для детекции зевания.
    Формула: ||top_lip - bottom_lip|| / ||left_corner - right_corner||
    """
    p_left = get_coords(landmarks, MOUTH_IDXS[0])
    p_right = get_coords(landmarks, MOUTH_IDXS[1])
    p_top = get_coords(landmarks, MOUTH_IDXS[2])
    p_bottom = get_coords(landmarks, MOUTH_IDXS[3])

    vertical = euclidean_distance(p_top, p_bottom)
    horizontal = euclidean_distance(p_left, p_right)

    return vertical / (horizontal + 1e-6)