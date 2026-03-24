import cv2
import numpy as np

mp = None
try:
    import mediapipe as mp
    mediapipe_available = True
except (ImportError, Exception) as e:
    print(f"MediaPipe not available: {e}")
    mediapipe_available = False

class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, 
                 refine_landmarks=True, min_detection_confidence=0.8,
                 min_tracking_confidence=0.8):
        if not mediapipe_available:
            self.face_mesh = None
            self.mp_face_mesh = None
            return
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame, draw=True):
        if self.face_mesh is None:
            return frame, None
            
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        results = self.face_mesh.process(rgb_image)
        
        rgb_image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks and draw:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        return image, results