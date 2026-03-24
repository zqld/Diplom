import numpy as np


class LandmarkUtils:
    """
    Универсальные функции для работы с MediaPipe landmarks.
    Поддерживает как старый API (0.9.x), так и новый (0.10.x).
    
    Старый API:  results.multi_face_landmarks[0].landmark[idx]
    Новый API:   results.face_landmarks[0][idx]
    """
    
    @staticmethod
    def is_new_api(results, landmark_type='face'):
        """
        Определить какой API используется.
        
        Args:
            results: Результат от MediaPipe
            landmark_type: 'face', 'hand', или 'pose'
            
        Returns:
            True если новый API, False если старый
        """
        if results is None:
            return False
            
        if landmark_type == 'face':
            if hasattr(results, 'face_landmarks'):
                return True
            if hasattr(results, 'multi_face_landmarks'):
                return False
                
        elif landmark_type == 'hand':
            if hasattr(results, 'hand_landmarks'):
                return True
            if hasattr(results, 'multi_hand_landmarks'):
                return False
                
        elif landmark_type == 'pose':
            if hasattr(results, 'pose_landmarks'):
                return True
            if hasattr(results, 'multi_pose_landmarks'):
                return False
        
        return False
    
    @staticmethod
    def get_face_landmarks(results):
        """
        Получить landmarks лица в универсальном формате.
        Возвращает список точек или None.
        
        Старый API: results.multi_face_landmarks[0].landmark
        Новый API: results.face_landmarks[0]
        """
        if results is None:
            return None
            
        try:
            if hasattr(results, 'face_landmarks') and results.face_landmarks:
                return results.face_landmarks[0]
            if hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
        except:
            pass
            
        return None
    
    @staticmethod
    def get_hand_landmarks(results):
        """
        Получить landmarks рук в универсальном формате.
        
        Старый API: results.multi_hand_landmarks
        Новый API: results.hand_landmarks
        """
        if results is None:
            return None
            
        try:
            if hasattr(results, 'hand_landmarks') and results.hand_landmarks:
                return results.hand_landmarks
            if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                return results.multi_hand_landmarks
        except:
            pass
            
        return None
    
    @staticmethod
    def get_pose_landmarks(results):
        """
        Получить landmarks позы в универсальном формате.
        
        Старый API: results.pose_landmarks[0].landmark
        Новый API: results.pose_landmarks[0]
        """
        if results is None:
            return None
            
        try:
            if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                return results.pose_landmarks[0]
            if hasattr(results, 'multi_pose_landmarks') and results.multi_pose_landmarks:
                return results.multi_pose_landmarks[0].landmark
        except:
            pass
            
        return None
    
    @staticmethod
    def get_point(landmarks, idx):
        """
        Получить точку landmarks по индексу.
        
        Работает с обоими форматами:
        - landmarks[idx].x, landmarks[idx].y (новый API)
        - landmarks[idx].x, landmarks[idx].y (старый API, уже извлечён landmark)
        """
        if landmarks is None:
            return None
            
        try:
            point = landmarks[idx]
            return (point.x, point.y, getattr(point, 'z', 0.0))
        except (IndexError, TypeError):
            return None
    
    @staticmethod
    def get_coords(landmarks, idx):
        """
        Получить координаты точки в виде numpy array [x, y].
        """
        point = LandmarkUtils.get_point(landmarks, idx)
        if point is None:
            return np.array([0.0, 0.0])
        return np.array([point[0], point[1]])
    
    @staticmethod
    def get_point_3d(landmarks, idx):
        """
        Получить 3D координаты точки.
        """
        point = LandmarkUtils.get_point(landmarks, idx)
        if point is None:
            return np.array([0.0, 0.0, 0.0])
        return np.array([point[0], point[1], point[2]])
    
    @staticmethod
    def get_all_coords(landmarks):
        """
        Получить все координаты landmarks.
        Возвращает массив формой (N, 2) или (N, 3).
        """
        if landmarks is None:
            return np.array([])
            
        try:
            has_z = len(landmarks) > 0 and hasattr(landmarks[0], 'z')
            
            if has_z:
                coords = np.array([[p.x, p.y, p.z] for p in landmarks])
            else:
                coords = np.array([[p.x, p.y] for p in landmarks])
                
            return coords
        except:
            return np.array([])
    
    @staticmethod
    def get_face_count(results):
        """Получить количество обнаруженных лиц."""
        if results is None:
            return 0
            
        try:
            if hasattr(results, 'face_landmarks') and results.face_landmarks:
                return len(results.face_landmarks)
            if hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
                return len(results.multi_face_landmarks)
        except:
            pass
            
        return 0
    
    @staticmethod
    def get_hand_count(results):
        """Получить количество обнаруженных рук."""
        if results is None:
            return 0
            
        try:
            if hasattr(results, 'hand_landmarks') and results.hand_landmarks:
                return len(results.hand_landmarks)
            if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                return len(results.multi_hand_landmarks)
        except:
            pass
            
        return 0
    
    @staticmethod
    def get_hand_handedness(results):
        """
        Получить информацию о том, какая рука.
        
        Старый API: results.multi_handedness[i].classification[0].label
        Новый API: results.handedness[i][0].category_name
        """
        if results is None:
            return []
            
        try:
            if hasattr(results, 'handedness') and results.handedness:
                handedness = []
                for h in results.handedness:
                    label = h[0].category_name if hasattr(h[0], 'category_name') else str(h[0])
                    handedness.append(label)
                return handedness
                
            if hasattr(results, 'multi_handedness') and results.multi_handedness:
                handedness = []
                for h in results.multi_handedness:
                    label = h.classification[0].label
                    handedness.append(label)
                return handedness
        except:
            pass
            
        return []
    
    @staticmethod
    def get_bounding_box(landmarks, frame_width, frame_height, padding=0.1):
        """
        Получить ограничивающую рамку для landmarks.
        
        Returns:
            (x_min, y_min, x_max, y_max) в пикселях
        """
        if landmarks is None:
            return (0, 0, frame_width, frame_height)
            
        try:
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            
            min_x = max(0, min(x_coords) - padding)
            max_x = min(1, max(x_coords) + padding)
            min_y = max(0, min(y_coords) - padding)
            max_y = min(1, max(y_coords) + padding)
            
            return (
                int(min_x * frame_width),
                int(min_y * frame_height),
                int(max_x * frame_width),
                int(max_y * frame_height)
            )
        except:
            return (0, 0, frame_width, frame_height)


landmarks_utils = LandmarkUtils()
