import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
from face_encoder import FaceEncoder
from webcam import VideoFinder

class FaceDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Person Finder Desktop GUI")
        self.setGeometry(100, 100, 800, 600)

        # GUI Elements
        self.video_label = QLabel(self)
        self.start_webcam_btn = QPushButton("Start Webcam", self)
        self.open_video_btn = QPushButton("Open Video File", self)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_webcam_btn)
        layout.addWidget(self.open_video_btn)
        self.setLayout(layout)

        # Signals
        self.start_webcam_btn.clicked.connect(self.start_webcam)
        self.open_video_btn.clicked.connect(self.open_video_file)

        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Video capture
        self.cap = None
        self.finder = None

        # Encode reference face
        encoder = FaceEncoder()
        self.reference_embedding = encoder.encode("data/reference.jpg")

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.finder = VideoFinder(self.reference_embedding, video_source=0)
        self.timer.start(30)  # 30 ms → ~33 FPS

    def open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.finder = VideoFinder(self.reference_embedding, video_source=file_path)
            self.timer.start(30)

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                self.cap.release()
                return

            frame = self.finder.detect_frame(frame)

            # Convert OpenCV frame to QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec())
