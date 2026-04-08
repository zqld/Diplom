import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        # 3D координаты стандартной модели лица (мм)
        # Точки: нос, подбородок, левый глаз, правый глаз, левый рот, правый рот
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Нос (tip)
            (0.0, -330.0, -65.0),        # Подбородок
            (-225.0, 170.0, -135.0),     # Левый глаз (внешний угол)
            (225.0, 170.0, -135.0),      # Правый глаз (внешний угол)
            (-150.0, -150.0, -125.0),    # Левый угол рта
            (150.0, -150.0, -125.0)      # Правый угол рта
        ])

    def get_pose(self, frame, landmarks):
        """
        Вычисляет углы наклона головы через PnP.
        Возвращает (pitch, yaw, roll) в градусах.

        pitch: наклон вперёд/назад (норма сидя: -5°...+5°)
        yaw:   поворот влево/вправо
        roll:  наклон головы к плечу
        """
        img_h, img_w, _ = frame.shape

        # MediaPipe FaceMesh indices → 2D pixel coords
        face_2d = []
        for idx in [1, 152, 33, 263, 61, 291]:
            x = int(landmarks.landmark[idx].x * img_w)
            y = int(landmarks.landmark[idx].y * img_h)
            face_2d.append([x, y])

        face_2d = np.array(face_2d, dtype=np.float64)

        # Матрица камеры (правильный порядок: cx = img_w/2, cy = img_h/2)
        focal_length = 1.0 * img_w
        cam_matrix = np.array([
            [focal_length, 0,          img_w / 2],
            [0,          focal_length, img_h / 2],
            [0,          0,            1]
        ])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # SolvePnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points, face_2d, cam_matrix, dist_matrix,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Rodrigues → rotation matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Rotation matrix → Euler angles (RQ decomposition)
        angles = cv2.RQDecomp3x3(rmat)[0]

        pitch = float(angles[0])
        yaw   = float(angles[1])
        roll  = float(angles[2])

        # Fold angles into [-90, 90]
        pitch = ((pitch + 90) % 180) - 90
        yaw   = ((yaw + 90) % 180) - 90
        roll  = ((roll + 90) % 180) - 90

        return pitch, yaw, roll