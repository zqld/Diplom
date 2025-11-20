import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, 
                 refine_landmarks=True, min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """
        Инициализация модели MediaPipe Face Mesh.
        
        :param refine_landmarks: Если True, добавляет точки для радужки глаз (важно для трекинга взгляда).
        :param max_num_faces: Максимальное количество лиц (нам нужно 1 для анализа пользователя).
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Инструменты для отрисовки (понадобятся для отладки/демонстрации)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame, draw=True):
        """
        Обрабатывает один кадр: находит лицо и (опционально) рисует сетку.
        
        :param frame: Кадр из OpenCV (BGR формат).
        :param draw: Рисовать ли сетку на изображении.
        :return: (image, results) - изображение с отрисовкой и сырые данные MediaPipe.
        """
        # 1. Конвертация BGR -> RGB (MediaPipe работает с RGB)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False  # Оптимизация производительности
        
        # 2. Поиск лица
        results = self.face_mesh.process(rgb_image)
        
        # 3. Возвращаем возможность записи и BGR формат
        rgb_image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # 4. Отрисовка сетки (визуализация для диплома)
        if results.multi_face_landmarks and draw:
            for face_landmarks in results.multi_face_landmarks:
                # Рисуем основную сетку лица (Tesselation)
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Рисуем контуры глаз и бровей (Contours)
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                # Рисуем радужку глаз (Iris) - важно для внимания!
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        return image, results