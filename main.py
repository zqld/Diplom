# main.py
import time
from collections import deque
import numpy as np
import cv2
from openvino.runtime import Core
from deepface import DeepFace
from plyer import notification
import matplotlib.pyplot as plt
import threading

# ---------- CONFIG ----------
MODEL_DIR = "models"
FD_MODEL = MODEL_DIR + "/face-detection-retail-0004.xml"
LM_MODEL = MODEL_DIR + "/facial-landmarks-35-adas-0002.xml"
EYE_MODEL = MODEL_DIR + "/open-closed-eye-0001.xml"

FACE_CONF_THRESH = 0.6
EYE_CLOSED_PROB_THRESH = 0.5
CLOSED_FRAMES_TO_ALERT = 15   # подряд закрытых кадров → предупреждение
EMOTION_CHECK_INTERVAL = 60   # секунд — запуск DeepFace каждые N сек
PLOT_WINDOW_SECONDS = 120     # окно для графика усталости
FPS_ESTIMATE = 20
# ----------------------------

ie = Core()

# Загрузка и компиляция моделей
fd_net = ie.read_model(FD_MODEL)
fd_comp = ie.compile_model(fd_net, "CPU")
fd_input_shape = fd_comp.inputs[0].shape  # [1,3,H,W] иногда

lm_net = ie.read_model(LM_MODEL)
lm_comp = ie.compile_model(lm_net, "CPU")
lm_input_shape = lm_comp.inputs[0].shape

eye_net = ie.read_model(EYE_MODEL)
eye_comp = ie.compile_model(eye_net, "CPU")
eye_input_shape = eye_comp.inputs[0].shape

# Подготовка видео
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# Для графика усталости
fatigue_history = deque(maxlen=PLOT_WINDOW_SECONDS * FPS_ESTIMATE)
time_history = deque(maxlen=PLOT_WINDOW_SECONDS * FPS_ESTIMATE)

# Счётчики
closed_eye_frame_count = 0
last_emotion_time = 0
last_emotion = "unknown"

# Простая notifier (Windows toast через plyer)
def send_notification(title, msg):
    try:
        notification.notify(title=title, message=msg, timeout=5)
    except Exception as e:
        print("Notify failed:", e)

# График в отдельном потоке
def plot_thread():
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,3))
    line, = ax.plot([], [])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, PLOT_WINDOW_SECONDS)
    ax.set_xlabel("seconds ago")
    ax.set_ylabel("fatigue_index")
    while True:
        if len(fatigue_history) > 1:
            y = list(fatigue_history)
            t = list(time_history)
            # преобразовать в "seconds ago" для X
            now = time.time()
            x = [now - ti for ti in t]
            # отображение последних PLOT_WINDOW_SECONDS секунд
            # перевернём и сделаем 0..PLOT_WINDOW_SECONDS
            x_plot = [PLOT_WINDOW_SECONDS - xv for xv in x]
            line.set_xdata(x_plot)
            line.set_ydata(y)
            ax.set_xlim(0, PLOT_WINDOW_SECONDS)
            ax.set_ylim(0, 1.05)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.5)

# Запускаем график
t_plot = threading.Thread(target=plot_thread, daemon=True)
t_plot.start()

def preprocess_for_model(img, target_shape):
    # target_shape like [1,3,H,W] (OpenVINO)
    _, c, h, w = target_shape
    img_resized = cv2.resize(img, (w, h))
    img_trans = img_resized.transpose((2,0,1))
    img_trans = np.expand_dims(img_trans, 0)
    return img_trans

print("Start camera. Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]
    orig = frame.copy()

    # --- Face detection ---
    fd_input = preprocess_for_model(frame, fd_comp.inputs[0].shape)
    fd_res = fd_comp([fd_input])[fd_comp.outputs[0]]
    # model format [1,1,N,7] typical -> [image_id, label, conf, x_min, y_min, x_max, y_max]
    detections = fd_res[0][0]
    best_box = None
    for det in detections:
        conf = float(det[2])
        if conf > FACE_CONF_THRESH:
            xmin = int(det[3] * W)
            ymin = int(det[4] * H)
            xmax = int(det[5] * W)
            ymax = int(det[6] * H)
            best_box = (xmin, ymin, xmax, ymax, conf)
            break

    fatigue_index = 0.0

    if best_box is not None:
        xmin, ymin, xmax, ymax, conf = best_box
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

        face_roi = orig[ymin:ymax, xmin:xmax]
        if face_roi.size == 0:
            continue

        # --- Landmarks ---
        lm_in = preprocess_for_model(face_roi, lm_comp.inputs[0].shape)
        lm_res = lm_comp([lm_in])[lm_comp.outputs[0]]
        # lm_res expected shape [1,70] or [1, X] -> sequence of normalized coords
        lm = lm_res.flatten()
        # every pair is x,y normalized to face_roi
        pts = []
        for i in range(0, len(lm), 2):
            x_rel = float(lm[i])
            y_rel = float(lm[i+1])
            x_pix = int(x_rel * (xmax - xmin)) + xmin
            y_pix = int(y_rel * (ymax - ymin)) + ymin
            pts.append((x_pix, y_pix))
            # optionally draw points:
            # cv2.circle(frame, (x_pix, y_pix), 1, (255,0,0), -1)

        # --- Heuristic: разделим landmarks на левую/правую часть лица для глаз ---
        # Это универсальный (не идеально точный) способ: берем точки левее центра лица -> левый набор
        cx = (xmin + xmax) // 2
        left_pts = [p for p in pts if p[0] < cx]
        right_pts = [p for p in pts if p[0] >= cx]

        def region_from_points(pts_group, pad=5):
            xs = [p[0] for p in pts_group]
            ys = [p[1] for p in pts_group]
            if not xs or not ys:
                return None
            x0, x1 = max(min(xs)-pad, 0), min(max(xs)+pad, W-1)
            y0, y1 = max(min(ys)-pad, 0), min(max(ys)+pad, H-1)
            return (x0, y0, x1, y1)

        left_eye_box = region_from_points(left_pts)
        right_eye_box = region_from_points(right_pts)

        prob_closed_values = []
        for box in (left_eye_box, right_eye_box):
            if box is None:
                continue
            x0,y0,x1,y1 = box
            eye_crop = orig[y0:y1, x0:x1]
            if eye_crop.size == 0:
                continue
            # preprocess and infer
            eye_in = preprocess_for_model(eye_crop, eye_comp.inputs[0].shape)
            eye_res = eye_comp([eye_in])[eye_comp.outputs[0]]
            # output usually probs for [open, closed] or [closed, open] — экспериментально выясни
            eye_scores = eye_res.flatten()
            # Простая логика: если два значения и second is 'closed' per model doc, check index
            if len(eye_scores) >= 2:
                # попробуем взять max-prob index; предполагаем индекс 1 == closed (проверь на своей модели!)
                prob_closed = float(eye_scores[1])
            else:
                prob_closed = float(eye_scores[0])
            prob_closed_values.append(prob_closed)
            # draw eye crop rect
            cv2.rectangle(frame, (x0,y0), (x1,y1), (255,0,0), 1)

        # если средняя вероятность закрытия высокая → считаем глаз закрытым
        if prob_closed_values:
            avg_closed_prob = sum(prob_closed_values) / len(prob_closed_values)
            if avg_closed_prob > EYE_CLOSED_PROB_THRESH:
                closed_eye_frame_count += 1
            else:
                closed_eye_frame_count = 0

            # индикатор усталости (0..1) — например, нормируем closed frames
            fatigue_index = min(1.0, closed_eye_frame_count / float(CLOSED_FRAMES_TO_ALERT))
            # подсказка на экране
            cv2.putText(frame, f"Fatigue idx: {fatigue_index:.2f}", (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if fatigue_index>0.5 else (0,255,0), 2)

            if closed_eye_frame_count >= CLOSED_FRAMES_TO_ALERT:
                send_notification("Пора отдохнуть", "Похоже, вы устали — сделайте паузу.")
                closed_eye_frame_count = 0

        # DeepFace периодически (эмоции)
        now = time.time()
        if now - last_emotion_time > EMOTION_CHECK_INTERVAL:
            # run DeepFace on face_roi (bgr->rgb)
            try:
                df_res = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                # DeepFace.analyze может возвращать список или dict
                if isinstance(df_res, list):
                    df_res = df_res[0]
                last_emotion = df_res.get('dominant_emotion', 'unknown')
                last_emotion_time = now
            except Exception as e:
                print("DeepFace error:", e)

        # Добавим текст эмоции
        cv2.putText(frame, f"Emotion: {last_emotion}", (xmin, ymax+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # добавить в историю для графика
    fatigue_history.append(fatigue_index)
    time_history.append(time.time())

    cv2.imshow("OpenVINO Fatigue + Emotions", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
