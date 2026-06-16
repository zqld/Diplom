# -*- mode: python ; coding: utf-8 -*-
#
# Сборка: pyinstaller neurofocus.spec
# После сборки: dist\NeuroFocus\NeuroFocus.exe
#

import glob
import os

SITE_PKGS = r'C:\Users\ZQL\AppData\Local\Programs\Python\Python311\Lib\site-packages'


import os
SITE_PKGS = r'C:\Users\ZQL\AppData\Local\Programs\Python\Python311\Lib\site-packages'

def collect_files(pattern, dest):
    return [(f, dest) for f in glob.glob(pattern, recursive=True)]


_vc_dlls = [
    os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', dll)
    for dll in ['MSVCP140.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'VCOMP140.DLL']
]

_mediapipe_binaries = [
    (os.path.join(SITE_PKGS, 'mediapipe', 'python', '_framework_bindings.cp311-win_amd64.pyd'),
     'mediapipe/python'),
    (os.path.join(SITE_PKGS, 'mediapipe', 'python', 'opencv_world3410.dll'),
     'mediapipe/python'),
    (os.path.join(SITE_PKGS, 'mediapipe', 'tasks', 'cc', 'metadata', 'python',
                   '_pywrap_metadata_version.cp311-win_amd64.pyd'),
     'mediapipe/tasks/cc/metadata/python'),
    (os.path.join(SITE_PKGS, 'mediapipe', 'tasks', 'python', 'metadata', 'flatbuffers_lib',
                   '_pywrap_flatbuffers.cp311-win_amd64.pyd'),
     'mediapipe/tasks/python/metadata/flatbuffers_lib'),
    # Копируем VC++ runtime DLLs рядом с _framework_bindings.pyd
    *[(dll, 'mediapipe/python') for dll in _vc_dlls if os.path.exists(dll)],
]


a = Analysis(
    ['main.py'],
    pathex=[SPECPATH],
    binaries=_mediapipe_binaries,
    datas=[
        *collect_files('models/*.keras', 'models'),
        *collect_files('models/*.hdf5', 'models'),
        *collect_files('models/*.task', 'models'),
        *collect_files(os.path.join(SITE_PKGS, 'cv2', 'data', 'haarcascade*.xml'), 'cv2/data'),
        ('config.json', '.'),
        ('assets/consent_text.txt', 'assets'),
    ],
    hiddenimports=[
        # ── MediaPipe ──────────────────────────────────────
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.face_mesh',
        'mediapipe.python.solutions.hands',
        'mediapipe.python.solutions.pose',
        'mediapipe.python.solutions.drawing_utils',
        'mediapipe.python.solutions.drawing_styles',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.core',
        'mediapipe.tasks.python.core.base_options',
        'mediapipe.tasks.python.vision',
        'mediapipe.tasks.python.vision.face_landmarker',
        'mediapipe.tasks.python.vision.hand_landmarker',
        'mediapipe.tasks.python.vision.pose_landmarker',
        'mediapipe.tasks.python.components',
        'mediapipe.tasks.python.components.containers',
        'mediapipe.tasks.python.components.containers.landmark',
        'mediapipe.tasks.python.components.containers.category',
        'mediapipe.framework',
        'mediapipe.framework.formats',
        'mediapipe.framework.formats.landmark_pb2',
        'mediapipe.calculators',
        # ── TensorFlow ─────────────────────────────────────
        'tensorflow',
        'tensorflow.lite',
        'tensorflow.python',
        'tensorflow.python.framework',
        'tensorflow.python.ops',
        'tensorflow.python.ops.array_ops',
        'tensorflow.python.ops.math_ops',
        'tensorflow.python.ops.nn_ops',
        'tensorflow.python.eager',
        'tensorflow.python.client',
        'tensorflow.python.saved_model',
        'tensorflow.python.keras',
        'tensorflow.python.keras.models',
        'tensorflow.python.keras.layers',
        'tensorflow.python.keras.optimizers',
        'tensorflow.python.keras.losses',
        'tensorflow.python.keras.metrics',
        'tensorflow.python.keras.utils',
        # ── Scikit-learn ───────────────────────────────────
        'sklearn',
        'sklearn.preprocessing',
        'sklearn.model_selection',
        'sklearn.metrics',
        'sklearn.base',
        'sklearn.utils',
        'sklearn.utils._typedefs',
        'sklearn.utils._cython_blas',
        'sklearn.utils.murmurhash',
        'sklearn.neighbors',
        'sklearn.neighbors._partition_nodes',
        'sklearn.svm',
        'sklearn.tree',
        'sklearn.ensemble',
        'sklearn.cluster',
        'sklearn.decomposition',
        'sklearn.pipeline',
        # ── OpenCV ─────────────────────────────────────────
        'cv2',
        'cv2.typing',
        # ── NumPy ──────────────────────────────────────────
        'numpy',
        'numpy.core',
        'numpy.random',
        'numpy.random._pickle',
        'numpy.random.mtrand',
        # ── PIL / Images ───────────────────────────────────
        'PIL',
        'PIL.Image',
        # ── Matplotlib ─────────────────────────────────────
        'matplotlib',
        'matplotlib.backends',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_agg',
        # ── Audio ──────────────────────────────────────────
        'pygame',
        'pygame.mixer',
        # ── Utilities ──────────────────────────────────────
        'pyautogui',
        'sqlalchemy',
        'sqlalchemy.ext',
        'sqlalchemy.ext.asyncio',
        'sqlalchemy.orm',
        'sqlalchemy.sql',
        'sqlalchemy.engine',
        'sqlalchemy.pool',
        'sqlalchemy.dialects',
        'sqlalchemy.dialects.sqlite',
        'pandas',
        'pandas._libs',
        'pandas._libs.tslibs',
        # ── Project modules ────────────────────────────────
        'build_utils',
        'src',
        'src.processors',
        'src.processors.face_processor',
        'src.processors.fatigue_processor',
        'src.processors.posture_processor',
        'src.processors.hand_processor',
        'src.face_core',
        'src.geometry',
        'src.emotion_detector',
        'src.fatigue_analyzer',
        'src.pose_estimator',
        'src.posture_analyzer',
        'src.hand_tracker',
        'src.gesture_controller',
        'src.database',
        'src.config_manager',
        'src.calibration_manager',
        'src.analytics',
        'src.data_exporter',
        'src.notifications',
        'src.sound_manager',
        'src.progress_tracker',
        'src.logger',
        # ── neurofocus ML ──────────────────────────────────
        'neurofocus',
        'neurofocus.ml',
        'neurofocus.ml.fatigue_classifier',
        'neurofocus.ml.posture_classifier',
        'neurofocus.ml.preprocessing',
        'neurofocus.ml.blink_tracker',
        'neurofocus.ml.microsleep_detector',
        'neurofocus.ml.temporal_features',
        'neurofocus.ml.user_profile',
        'neurofocus.ml.threshold_adapter',
        'neurofocus.ml.online_learner',
        'neurofocus.ml.ml_coordinator',
        'neurofocus.ml.posture_data_generator',
        'neurofocus.detectors',
        'neurofocus.detectors.pose_detector',
        # ── UI ─────────────────────────────────────────────
        'ui',
        'ui.calibration',
        'ui.settings',
        'ui.stats',
        'ui.progress',
        'ui.pomodoro',
        'ui.help',
        'gui_about',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_dll.py'],
    excludes=[
        'pytest',
        'tkinter',
        'jupyter',
        'IPython',

    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuroFocus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'MSVCP140.dll',
        'VCRUNTIME140.dll',
        'VCRUNTIME140_1.dll',
        'VCOMP140.DLL',
        '_framework_bindings*',
    ],
    name='NeuroFocus',
)

