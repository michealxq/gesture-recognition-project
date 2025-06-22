
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import csv
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import vlc
from collections import deque, Counter
from PyQt5 import QtCore, QtGui, QtWidgets

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input  # type: ignore

# === SETTINGS ===
VIDEO_DIR        = "videos"
KERAS_MODEL_PATH = os.path.join("model", "mobilenetv3_hagrid_finetuned.keras")
LABEL_CSV        = os.path.join("model", "gesture_labels.csv")
CAM_WIDTH        = 320
CAM_HEIGHT       = 240
TIMER_INTERVAL   = 80      # ms (~12 FPS)
VOLUME_STEP      = 10      # percent
SKIP_MS          = 10_000  # 10 seconds in ms
CONF_THRESH      = 0.4     # minimum softmax confidence
HOLD_FRAMES      = 6       # how many consecutive frames to hold gesture

GESTURE_COMMANDS = {
    "palm":             "play",
    "fist":             "pause",
    "call":             "next",
    "rock":             "prev",
    "like":             "vol_up",
    "dislike":          "vol_down",
    "two_up":           "skip_fwd",
    "two_up_inverted":  "skip_back",
    "stop":             "quit",
}

# === GESTURE RECOGNIZER ===
class GestureRecognizer:
    def __init__(self,
                 model_path=KERAS_MODEL_PATH,
                 label_csv=LABEL_CSV,
                 conf_thresh=CONF_THRESH,
                 pad=0.5):
        # load labels
        with open(label_csv, newline="") as f:
            self.labels = [r[0] for r in csv.reader(f) if r]

        # load Keras model
        self.model = tf.keras.models.load_model(model_path)
        _, h, w, _ = self.model.input_shape
        self.H, self.W = h, w

        self.conf_thresh = conf_thresh
        self.pad = pad

        # MediaPipe for hand detection
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def predict(self, frame):
        """
        Returns a tuple (label, confidence, crop_box)
        or (None, 0.0, None) if no confident prediction.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, 0.0, None

        # get pixel coordinates of hand landmarks
        h0, w0, _ = frame.shape
        pts = [(int(lm.x * w0), int(lm.y * h0))
               for lm in res.multi_hand_landmarks[0].landmark]
        xs, ys = zip(*pts)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        # padded square crop
        pad_px = int(side * self.pad)
        half = side // 2 + pad_px
        x0_ = max(0, cx - half)
        x1_ = min(w0, cx + half)
        y0_ = max(0, cy - half)
        y1_ = min(h0, cy + half)
        crop = frame[y0_:y1_, x0_:x1_]
        if crop.size == 0:
            return None, 0.0, None

        # remember for drawing
        crop_box = (x0_, y0_, x1_, y1_)

        # preprocess for model
        img = cv2.resize(crop, (self.W, self.H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = preprocess_input(img)              # [-1,1] for MobileNetV3
        inp = np.expand_dims(img, 0)

        out = self.model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(out))
        conf = float(out[idx])
        label = self.labels[idx] if conf > self.conf_thresh else None

        return label, conf, crop_box

# === MAIN WINDOW ===
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture‐Controlled Media Player")
        self.setFixedSize(1000, 600)

        # video list
        self.videos = sorted(
            os.path.join(VIDEO_DIR, f)
            for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(".mp4")
        )
        if not self.videos:
            QtWidgets.QMessageBox.critical(self, "Error", "No .mp4 files in videos/")
            sys.exit(1)

        # VLC player
        self.player = vlc.Instance("--quiet").media_player_new()

        # recognizer
        self.recognizer = GestureRecognizer()
        self.hold_counts = {g: 0 for g in GESTURE_COMMANDS}
        self.last_cmd = None

        # webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        # load icons (optional)
        self.icons = {}
        icon_path = "icons"
        for act in GESTURE_COMMANDS.values():
            fn = os.path.join(icon_path, f"{act}.png")
            if os.path.isfile(fn):
                self.icons[act] = QtGui.QPixmap(fn)

        self._build_ui()
        self._start_timers()

        # start on first video
        self.idx = 0
        self._load(self.idx)
        self.player.play()

    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        main_layout = QtWidgets.QHBoxLayout(w)

        # === LEFT PANEL ===
        left_panel = QtWidgets.QVBoxLayout()

        # --- Status Section ---
        self.lbl_gst = QtWidgets.QLabel("Gesture: –")
        self.lbl_gst.setStyleSheet("font:14pt; color:blue;")
        left_panel.addWidget(self.lbl_gst)

        self.conf_bar = QtWidgets.QProgressBar()
        self.conf_bar.setRange(0, 100)
        left_panel.addWidget(self.conf_bar)

        self.lbl_vol = QtWidgets.QLabel("Vol: 50%")
        self.lbl_vol.setStyleSheet("font:14pt; color:green;")
        left_panel.addWidget(self.lbl_vol)

        left_panel.addWidget(QtWidgets.QLabel("Crop padding"))
        self.pad_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pad_slider.setRange(0, 100)
        self.pad_slider.setValue(int(self.recognizer.pad * 100))
        self.pad_slider.valueChanged.connect(
            lambda v: setattr(self.recognizer, "pad", v / 100)
        )
        left_panel.addWidget(self.pad_slider)

        # --- Controls Info Section ---
        
        # Emojis for gesture and actions
        gesture_display = {
            "palm":             "🖐️ Palm              ",
            "fist":             "✊ Fist              ",
            "call":             "🤙 Call              ",
            "rock":             "🤘 Rock              ",
            "like":             "👍 Like              ",
            "dislike":          "👎 Dislike           ",
            "two_up":           "✌️ Two Up            ",
            "two_up_inverted":  "✌️ Two reversed",
            "stop":             "✋ Stop              "
        }

        action_emojis = {
            "play":        "▶️",
            "pause":       "⏸️",
            "next":        "⏭️",
            "prev":        "⏮️",
            "vol_up":      "🔊",
            "vol_down":    "🔉",
            "skip_fwd":    "⏩",
            "skip_back":   "⏪",
            "quit":        "🛑",
        }

        # Format rows with padded columns
        rows = []
        for g_key, action in GESTURE_COMMANDS.items():
            gesture_name = gesture_display.get(g_key, g_key)
            action_icon = action_emojis.get(action, "")
            rows.append(f"{gesture_name:<40} | {action:<12} {action_icon}")

        # Build final text
        controls_text = "🧠 Hand Gestures and Controls:\n\n" + "\n".join(rows)

        self.lbl_controls = QtWidgets.QLabel(controls_text)
        self.lbl_controls.setStyleSheet("font:10pt; font-family: monospace;")
        self.lbl_controls.setWordWrap(True)

        self.lbl_controls.setStyleSheet("font:10pt;")
        self.lbl_controls.setWordWrap(True)
        left_panel.addWidget(self.lbl_controls)

        left_panel.addStretch()
        main_layout.addLayout(left_panel)

        # === RIGHT PANEL ===
        right_panel = QtWidgets.QVBoxLayout()

        # --- Camera Feed ---
        self.lbl_cam = QtWidgets.QLabel()
        self.lbl_cam.setFixedSize(CAM_WIDTH, CAM_HEIGHT)
        right_panel.addWidget(self.lbl_cam, alignment=QtCore.Qt.AlignCenter)

        # --- Video Player ---
        self.vframe = QtWidgets.QFrame()
        self.vframe.setFixedSize(640, 360)
        right_panel.addWidget(self.vframe, alignment=QtCore.Qt.AlignCenter)

        self.lbl_vid = QtWidgets.QLabel()
        self.lbl_vid.setStyleSheet("font:16pt;")
        right_panel.addWidget(self.lbl_vid, alignment=QtCore.Qt.AlignCenter)

        if sys.platform.startswith("win"):
            self.player.set_hwnd(int(self.vframe.winId()))
        else:
            self.player.set_xwindow(self.vframe.winId())

        main_layout.addLayout(right_panel)


    def _start_timers(self):
        self.tcam = QtCore.QTimer(self)
        self.tcam.timeout.connect(self._update_frame)
        self.tcam.start(TIMER_INTERVAL)

        self.tv = QtCore.QTimer(self)
        self.tv.timeout.connect(self._update_video_label)
        self.tv.start(200)

    def _load(self, i):
        m = vlc.Instance("--quiet").media_new(self.videos[i])
        self.player.set_media(m)
        self.player.audio_set_volume(50)
        self.lbl_vol.setText("Vol: 50%")
        self.lbl_vid.setText(f"Video: {os.path.basename(self.videos[i])}")

    def _update_video_label(self):
        self.lbl_vid.setText(f"Video: {os.path.basename(self.videos[self.idx])}")

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)

        # midline_y = frame.shape[0] // 2
        # cv2.line(frame, (0, midline_y), (frame.shape[1], midline_y), (0, 0, 255), 2)

        # predict
        gst, conf, box = self.recognizer.predict(frame)
        # gst, conf, box = None, 0.0, None
        # pred_label, pred_conf, pred_box = self.recognizer.predict(frame)

        # # Only accept predictions above the midline
        # if pred_box:
        #     _, y0, _, y1 = pred_box
        #     midline_y = frame.shape[0] // 2
        #     if y1 < midline_y:  # whole hand above the line
        #         gst, conf, box = pred_label, pred_conf, pred_box

        # draw crop box
        if box:
            cv2.rectangle(frame,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0,255,0), 2)

        # update confidence bar
        self.conf_bar.setValue(int(conf * 100))

        # hold-to-activate logic
        if gst in GESTURE_COMMANDS:
            self.hold_counts[gst] += 1
            if self.hold_counts[gst] >= HOLD_FRAMES and gst != self.last_cmd:
                self._apply(GESTURE_COMMANDS[gst])
                self.last_cmd = gst
        # reset others
        for g in list(self.hold_counts):
            if g != gst:
                self.hold_counts[g] = 0

        self.lbl_gst.setText(f"Gesture: {gst or '–'}")

        # show webcam with box
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h0, w0, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w0, h0, w0*3, QtGui.QImage.Format_RGB888)
        self.lbl_cam.setPixmap(QtGui.QPixmap.fromImage(img))

    def _flash_icon(self, action):
        pix = self.icons.get(action)
        if pix:
            lbl = QtWidgets.QLabel(self.vframe)
            lbl.setPixmap(pix)
            lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            lbl.move((640-pix.width())//2, (360-pix.height())//2)
            lbl.show()
            QtCore.QTimer.singleShot(700, lbl.deleteLater)

    def _apply(self, action):
        if action == "play":
            self.player.play()
        elif action == "pause":
            self.player.pause()
        elif action == "next":
            self.idx = (self.idx + 1) % len(self.videos)
            self._load(self.idx)
            self.player.play()
        elif action == "prev":
            self.idx = (self.idx - 1) % len(self.videos)
            self._load(self.idx)
            self.player.play()
        elif action == "vol_up":
            v = min(100, self.player.audio_get_volume() + VOLUME_STEP)
            self.player.audio_set_volume(v)
            self.lbl_vol.setText(f"Vol: {v}%")
        elif action == "vol_down":
            v = max(0, self.player.audio_get_volume() - VOLUME_STEP)
            self.player.audio_set_volume(v)
            self.lbl_vol.setText(f"Vol: {v}%")
        elif action == "skip_fwd":
            t = self.player.get_time() or 0
            self.player.set_time(min(self.player.get_length(), t + SKIP_MS))
        elif action == "skip_back":
            t = self.player.get_time() or 0
            self.player.set_time(max(0, t - SKIP_MS))
        elif action == "quit":
            self.close()

        # flash corresponding icon
        self._flash_icon(action)

    def closeEvent(self, event):
        self.cap.release()
        self.player.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
