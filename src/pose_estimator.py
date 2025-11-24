import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        # 3D координаты стандартной модели лица
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Нос
            (0.0, -330.0, -65.0),        # Подбородок
            (-225.0, 170.0, -135.0),     # Левый глаз
            (225.0, 170.0, -135.0),      # Правый глаз
            (-150.0, -150.0, -125.0),    # Левый рот
            (150.0, -150.0, -125.0)      # Правый рот
        ])

    def get_pose(self, frame, landmarks):
        """
        Вычисляет углы наклона головы.
        Возвращает (pitch, yaw, roll) в нормальных градусах.
        """
        img_h, img_w, _ = frame.shape
        
        face_2d = []
        for idx in [1, 152, 33, 263, 61, 291]:
            x, y = int(landmarks.landmark[idx].x * img_w), int(landmarks.landmark[idx].y * img_h)
            face_2d.append([x, y])
            
        face_2d = np.array(face_2d, dtype=np.float64)

        # Матрица камеры
        focal_length = 1 * img_w
        cam_matrix = np.array([ 
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # PnP
        success, rot_vec, trans_vec = cv2.solvePnP(self.model_points, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)

        # Получаем матрицу вращения
        rmat, jac = cv2.Rodrigues(rot_vec)
        
        # Получаем углы Эйлера
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # --- ИСПРАВЛЕНИЕ КООРДИНАТ ---
        # Если значения "скачут" вокруг 180 (например -170... 170), сдвигаем их
        if pitch < -90:
            pitch += 180
        elif pitch > 90:
            pitch -= 180
            
        # Иногда ось инвертирована (верх-низ), умножаем на скорректированный коэффициент
        # Для большинства веб-камер это приводит ось в норму
        return pitch * 1.5, yaw, roll