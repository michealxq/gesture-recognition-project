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
        with open(label_csv, newline="") as f:
            self.labels = [r[0] for r in csv.reader(f) if r]

        self.model = tf.keras.models.load_model(model_path)
        _, h, w, _ = self.model.input_shape
        self.H, self.W = h, w

        self.conf_thresh = conf_thresh
        self.pad = pad

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, 0.0, None

        h0, w0, _ = frame.shape
        pts = [(int(lm.x * w0), int(lm.y * h0))
               for lm in res.multi_hand_landmarks[0].landmark]
        xs, ys = zip(*pts)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        pad_px = int(side * self.pad)
        half = side // 2 + pad_px
        x0_ = max(0, cx - half)
        x1_ = min(w0, cx + half)
        y0_ = max(0, cy - half)
        y1_ = min(h0, cy + half)
        crop = frame[y0_:y1_, x0_:x1_]
        if crop.size == 0:
            return None, 0.0, None

        crop_box = (x0_, y0_, x1_, y1_)

        img = cv2.resize(crop, (self.W, self.H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = preprocess_input(img)
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
        self.setWindowTitle("Gesture-Controlled Media Player")
        self.showMaximized()
        # self.showFullScreen()


        # Check camera availability
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Could not open webcam!")
            sys.exit(1)

        self.videos = sorted(
            os.path.join(VIDEO_DIR, f)
            for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(".mp4")
        )
        if not self.videos:
            QtWidgets.QMessageBox.critical(self, "Error", "No .mp4 files in videos/")
            sys.exit(1)

        self.player = vlc.Instance("--quiet").media_player_new()
        self.recognizer = GestureRecognizer()
        self.hold_counts = {g: 0 for g in GESTURE_COMMANDS}
        self.last_cmd = None

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        self.icons = {}
        icon_path = "icons"
        for act in GESTURE_COMMANDS.values():
            fn = os.path.join(icon_path, f"{act}.png")
            if os.path.isfile(fn):
                self.icons[act] = QtGui.QPixmap(fn)

        self._build_ui()
        self._start_timers()

        self.idx = 0
        self._load(self.idx)
        self.player.play()

        self._apply_styles()
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.showNormal()
    
    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QLabel {
                color: #000000;
            }
            QGroupBox {
                border: 1px solid #aaa;
                margin-top: 10px;
                font-weight: bold;
                color: #000000;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QProgressBar {
                height: 18px;
                text-align: center;
                border: 1px solid #888;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #cccccc;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4caf50;
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -3px 0;
            }
            QScrollArea {
                background-color: #ffffff;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #ffffff;
            }
            QWidget {
                background-color: #ffffff;
                color: #000000;
            }
        """)

    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QHBoxLayout(w)

        left_panel = QtWidgets.QVBoxLayout()

        # Status group
        grp_status = QtWidgets.QGroupBox("Gesture Status")
        status_layout = QtWidgets.QVBoxLayout(grp_status)

        self.lbl_gst = QtWidgets.QLabel("Gesture: –")
        self.lbl_gst.setStyleSheet("font: bold 16pt; color: #4caf50;")
        status_layout.addWidget(self.lbl_gst)

        self.conf_bar = QtWidgets.QProgressBar()
        self.conf_bar.setRange(0, 100)
        status_layout.addWidget(self.conf_bar)

        self.lbl_vol = QtWidgets.QLabel("Vol: 50%")
        self.lbl_vol.setStyleSheet("font: bold 16pt; color: #f9a825;")
        status_layout.addWidget(self.lbl_vol)

        self.lbl_feedback = QtWidgets.QLabel("")
        self.lbl_feedback.setStyleSheet("font: bold 14pt; color: #ffffff;")
        status_layout.addWidget(self.lbl_feedback)

        left_panel.addWidget(grp_status)

        # Crop slider
        grp_crop = QtWidgets.QGroupBox("Crop Padding")
        crop_layout = QtWidgets.QVBoxLayout(grp_crop)
        self.pad_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pad_slider.setRange(0, 100)
        self.pad_slider.setValue(int(self.recognizer.pad * 100))
        self.pad_slider.valueChanged.connect(
            lambda v: setattr(self.recognizer, "pad", v / 100)
        )
        crop_layout.addWidget(self.pad_slider)
        left_panel.addWidget(grp_crop)

        # Controls group
        grp_controls = QtWidgets.QGroupBox("Hand Gesture Controls")
        controls_layout = QtWidgets.QVBoxLayout(grp_controls)

        gesture_display = {
            "palm": "🖐 Palm",
            "fist": "✊ Fist",
            "call": "🤙 Call",
            "rock": "🤘 Rock",
            "like": "👍 Like",
            "dislike": "👎 Dislike",
            "two_up": "✌️ Two Up",
            "two_up_inverted": "✌️ Two Reversed",
            "stop": "✋ Stop",
        }

        action_emojis = {
            "play": "▶️",
            "pause": "⏸️",
            "next": "⏭️",
            "prev": "⏮️",
            "vol_up": "🔊",
            "vol_down": "🔉",
            "skip_fwd": "⏩",
            "skip_back": "⏪",
            "quit": "🛑",
        }

        # Create inner widget for scroll area
        scroll_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(scroll_widget)

        for row, (g_key, action) in enumerate(GESTURE_COMMANDS.items()):
            gesture_label = QtWidgets.QLabel(gesture_display.get(g_key, g_key))
            gesture_label.setStyleSheet("font: 12pt; color: #000000;")

            action_label = QtWidgets.QLabel(action)
            action_label.setStyleSheet("font: 12pt; color: #000000;")

            icon_text = action_emojis.get(action, "")
            action_icon = QtWidgets.QLabel(icon_text)
            action_icon.setStyleSheet("font: 12pt;color: #000000")

            grid.addWidget(gesture_label, row, 0)
            grid.addWidget(action_label, row, 1)
            grid.addWidget(action_icon, row, 2)

        # Tweak column spacing for nice alignment
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_widget)
        scroll.setStyleSheet("background-color: #ffffff; border: none;")


        controls_layout.addWidget(scroll)

        left_panel.addWidget(grp_controls)

        layout.addLayout(left_panel)  

        # RIGHT PANEL
        right_panel = QtWidgets.QVBoxLayout()

        self.lbl_cam = QtWidgets.QLabel()
        self.lbl_cam.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        right_panel.addWidget(self.lbl_cam, alignment=QtCore.Qt.AlignCenter)

        self.vframe = QtWidgets.QFrame()
        self.vframe.setMinimumSize(640, 360)
        self.vframe.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                          QtWidgets.QSizePolicy.Expanding)
        self.vframe.setStyleSheet("background-color: black; border: 1px solid #555;")
        video_container = QtWidgets.QHBoxLayout()
        video_container.addStretch()
        video_container.addWidget(self.vframe)
        video_container.addStretch()
        right_panel.addLayout(video_container)

        self.lbl_vid = QtWidgets.QLabel()
        self.lbl_vid.setWordWrap(True)
        self.lbl_vid.setStyleSheet("font: bold 14pt; color: #000000;")
        right_panel.addWidget(self.lbl_vid, alignment=QtCore.Qt.AlignCenter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        right_panel.addWidget(self.slider)

        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font: 12pt;")
        right_panel.addWidget(self.lbl_time, alignment=QtCore.Qt.AlignCenter)

        if sys.platform.startswith("win"):
            self.player.set_hwnd(int(self.vframe.winId()))
        else:
            self.player.set_xwindow(self.vframe.winId())

        layout.addLayout(right_panel)

    def _start_timers(self):
        self.tcam = QtCore.QTimer(self)
        self.tcam.timeout.connect(self._update_frame)
        self.tcam.start(TIMER_INTERVAL)

        self.tv = QtCore.QTimer(self)
        self.tv.timeout.connect(self._update_video_label)
        self.tv.start(300)

    def _load(self, i):
        m = vlc.Instance("--quiet").media_new(self.videos[i])
        self.player.set_media(m)
        self.player.audio_set_volume(50)
        self.lbl_vol.setText("Vol: 50%")
        self.lbl_vid.setText(f"Video: {os.path.basename(self.videos[i])}")

    def _update_video_label(self):
        self.lbl_vid.setText(f"Video: {os.path.basename(self.videos[self.idx])}")

        length = self.player.get_length()
        pos = self.player.get_time()
        if length > 0:
            secs_total = length // 1000
            secs_pos = pos // 1000
            self.lbl_time.setText(
                f"{secs_pos//60:02d}:{secs_pos%60:02d} / {secs_total//60:02d}:{secs_total%60:02d}"
            )
            self.slider.setValue(int(pos / length * 1000))
        else:
            self.lbl_time.setText("00:00 / 00:00")

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)

        gst, conf, box = self.recognizer.predict(frame)

        if box:
            cv2.rectangle(frame,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 255, 0), 2)

        self.conf_bar.setValue(int(conf * 100))

        if gst in GESTURE_COMMANDS:
            self.hold_counts[gst] += 1
            if self.hold_counts[gst] >= HOLD_FRAMES and gst != self.last_cmd:
                self._apply(GESTURE_COMMANDS[gst])
                self.last_cmd = gst
        for g in list(self.hold_counts):
            if g != gst:
                self.hold_counts[g] = 0

        self.lbl_gst.setText(f"Gesture: {gst or '–'}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h0, w0, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w0, h0, w0*3, QtGui.QImage.Format_RGB888)
        self.lbl_cam.setPixmap(QtGui.QPixmap.fromImage(img))

    def _flash_icon(self, action):
        pix = self.icons.get(action)
        if pix:
            lbl = QtWidgets.QLabel(self.vframe)
            lbl.setPixmap(pix.scaled(64, 64, QtCore.Qt.KeepAspectRatio))
            lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            lbl.move((640 - 64)//2, (360 - 64)//2)
            lbl.show()
            QtCore.QTimer.singleShot(700, lbl.deleteLater)

    def _apply(self, action):
        self.lbl_feedback.setText(f"Action: {action.upper()}")
        QtCore.QTimer.singleShot(1500, lambda: self.lbl_feedback.clear())

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
