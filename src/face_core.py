import cv2
import numpy as np
import sys
import os

# Для PyInstaller frozen-режима: добавляем _internal в DLL search path
if getattr(sys, 'frozen', False):
    try:
        os.add_dll_directory(sys._MEIPASS)
        mp_pyd_dir = os.path.join(sys._MEIPASS, 'mediapipe', 'python')
        if os.path.isdir(mp_pyd_dir):
            os.add_dll_directory(mp_pyd_dir)
    except Exception:
        pass

mp = None
try:
    import mediapipe as mp
    mediapipe_available = True
except (ImportError, Exception) as e:
    print(f"MediaPipe not available: {e}")
    mediapipe_available = False


class _FallbackFaceDetector:
    """OpenCV Haar Cascade fallback when MediaPipe is unavailable."""

    def __init__(self):
        self._cascade = None
        try:
            cascade_paths = [
                os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'),
                os.path.join(os.path.dirname(cv2.__file__), 'data',
                             'haarcascade_frontalface_default.xml'),
            ]
            for p in cascade_paths:
                if os.path.exists(p):
                    self._cascade = cv2.CascadeClassifier(p)
                    if not self._cascade.empty():
                        break
                    self._cascade = None
        except Exception:
            pass
        self._prev_bbox = None

    def _build_mock_results(self, frame, bbox):
        """Create a mock MediaPipe results object from a face bounding box.
        Key landmarks produce EAR~0.3, MAR~0.15 for an open eye/mouth.
        """
        h, w = frame.shape[:2]
        fx, fy, fw, fh = bbox
        # face center
        cx_n = (fx + fw / 2) / w
        cy_n = (fy + fh / 2) / h

        class MockLandmark:
            def __init__(self, x, y, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        class MockLandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks

        def nrm(px, py):
            """Convert pixel coords to normalized 0-1."""
            return px / w, py / h

        def pt(fx_frac, fy_frac):
            """Point at (fx + fx_frac*fw, fy + fy_frac*fh) in normalized."""
            return nrm(fx + fx_frac * fw, fy + fy_frac * fh)

        # Fill all 468 with a default at face center
        lm = [MockLandmark(cx_n, cy_n)] * 468

        # --- Left eye (person's left → image right side) ---
        # Eye box: 55%-70% face width, 36%-41% face height
        le_x33, le_y = pt(0.55, 0.38)   # 33 inner (nose side)
        le_x133 = (fx + 0.70 * fw) / w   # 133 outer (ear side)
        ey_v = 0.008  # vertical half-opening → EAR ≈ (0.016)/(2*0.045) = 0.18
        lx = (fx + 0.62 * fw) / w  # x center of left eye for vertical lands
        lm[33] = MockLandmark(le_x33, le_y)
        lm[133] = MockLandmark(le_x133, le_y)
        lm[160] = MockLandmark(lx, le_y - ey_v)
        lm[144] = MockLandmark(lx, le_y + ey_v)
        lm[158] = MockLandmark(lx, le_y - ey_v)
        lm[153] = MockLandmark(lx, le_y + ey_v)

        # --- Right eye (person's right → image left side) ---
        re_x362, re_y = pt(0.45, 0.38)  # 362 inner (nose side)
        re_x263 = (fx + 0.30 * fw) / w  # 263 outer (ear side)
        rx = (fx + 0.38 * fw) / w
        lm[362] = MockLandmark(re_x362, re_y)
        lm[263] = MockLandmark(re_x263, re_y)
        lm[385] = MockLandmark(rx, re_y - ey_v)
        lm[380] = MockLandmark(rx, re_y + ey_v)
        lm[387] = MockLandmark(rx, re_y - ey_v)
        lm[373] = MockLandmark(rx, re_y + ey_v)

        # --- Nose tip ---
        lm[1] = MockLandmark(*pt(0.50, 0.58))

        # --- Mouth ---
        my = (fy + 0.68 * fh) / h
        mh = 0.006  # mouth vertical half-opening → MAR ≈ 0.012/0.075 = 0.16
        lm[61] = MockLandmark(*pt(0.40, 0.68))   # left corner
        lm[291] = MockLandmark(*pt(0.60, 0.68))  # right corner
        lm[13] = MockLandmark(cx_n, my + mh)      # bottom lip
        lm[14] = MockLandmark(cx_n, my - mh)      # top lip

        # --- Chin & forehead & ears ---
        lm[152] = MockLandmark(*pt(0.50, 0.95))  # chin
        lm[10] = MockLandmark(*pt(0.50, 0.05))   # forehead
        lm[234] = MockLandmark(*pt(0.05, 0.40))  # left ear (image left)
        lm[454] = MockLandmark(*pt(0.95, 0.40))  # right ear (image right)

        class MockResults:
            def __init__(self, landmark_list):
                self.multi_face_landmarks = [landmark_list]

        return MockResults(MockLandmarkList(lm))

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            self._prev_bbox = None
            return None
        bbox = tuple(faces[0])
        self._prev_bbox = bbox
        return self._build_mock_results(frame, bbox)

    def draw_bbox(self, frame, bbox):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, 
                 refine_landmarks=True, min_detection_confidence=0.8,
                 min_tracking_confidence=0.8):
        self._fallback = None

        if not mediapipe_available:
            self.face_mesh = None
            self.mp_face_mesh = None
            self._fallback = _FallbackFaceDetector()
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
        # Fallback when MediaPipe is not available
        if self.face_mesh is None:
            if self._fallback is not None:
                results = self._fallback.detect(frame)
                if results is not None and draw:
                    # draw bounding box on frame
                    self._fallback.draw_bbox(
                        frame, self._fallback._prev_bbox)
                return frame, results
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