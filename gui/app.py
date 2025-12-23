import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider, QMessageBox
)
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPixmap

from face_encoder import FaceEncoder
from webcam import VideoFinder
from backend.video_feed import VideoFeed


# class VideoWorker(QThread):
#     def __init__(self, ref_image, source, threshold):
#         super().__init__()
#         self.ref_image = ref_image
#         self.source = source
#         self.threshold = threshold
#         self.feed = None

#     def run(self):
#         encoder = FaceEncoder()
#         embedding = encoder.encode(self.ref_image)

#         finder = VideoFinder(
#             reference_embedding=embedding,
#             video_source=self.source,
#             threshold=self.threshold,
#         )

#         self.feed = VideoFeed(finder)
#         self.feed.start()

#     def stop(self):
#         if self.feed:
#             self.feed.stop()

class VideoWorker(QThread):
    def __init__(self, ref_image, source, threshold):
        super().__init__()
        self.ref_image = ref_image
        self.source = source
        self.threshold = threshold
        self.feed = None

    def run(self):
        encoder = FaceEncoder()
        embedding = encoder.encode(self.ref_image)

        finder = VideoFinder(
            reference_embedding=embedding,
            video_source=self.source,
            threshold=self.threshold,
        )

        self.feed = VideoFeed(finder)
        self.feed.start()

    def stop(self):
        if self.feed:
            self.feed.stop()
        self.quit()        # tell Qt to stop thread loop
        self.wait(100)     # short wait, DO NOT block GUI




class FindPersonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Find Person System")
        self.setFixedSize(500, 520)

        self.ref_image = None
        self.video_path = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Person Finder")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        # Reference image preview
        self.image_preview = QLabel("No Image")
        self.image_preview.setFixedSize(200, 200)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_preview, alignment=Qt.AlignCenter)

        btn_ref = QPushButton("Upload Reference Image")
        btn_ref.clicked.connect(self.upload_reference)
        layout.addWidget(btn_ref)

        # Threshold slider
        self.threshold_label = QLabel("Threshold: 0.50")
        layout.addWidget(self.threshold_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(30, 80)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"Threshold: {v/100:.2f}")
        )
        layout.addWidget(self.slider)

        # Buttons
        btn_live = QPushButton("Start Live Webcam")
        btn_live.clicked.connect(self.start_webcam)
        layout.addWidget(btn_live)

        btn_video = QPushButton("Select Recorded Video")
        btn_video.clicked.connect(self.select_video)
        layout.addWidget(btn_video)

        btn_start_video = QPushButton("Start Recorded Video")
        btn_start_video.clicked.connect(self.start_video)
        layout.addWidget(btn_start_video)

        btn_stop = QPushButton("STOP")
        btn_stop.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        btn_stop.clicked.connect(self.stop_feed)
        layout.addWidget(btn_stop)

        self.setLayout(layout)

    # ---------- Actions ----------

    def upload_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png)")
        if path:
            self.ref_image = path
            pix = QPixmap(path).scaled(200, 200, Qt.KeepAspectRatio)
            self.image_preview.setPixmap(pix)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi)")
        if path:
            self.video_path = path

    def start_webcam(self):
        self._start_feed(0)

    def start_video(self):
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Select a video first.")
            return
        self._start_feed(self.video_path)

    def _start_feed(self, source):
        if not self.ref_image:
            QMessageBox.warning(self, "Error", "Upload reference image first.")
            return

        self.worker = VideoWorker(
            self.ref_image,
            source,
            self.slider.value() / 100,
        )
        self.worker.start()

    def stop_feed(self):
        if self.worker:
            self.worker.stop()


def main():
    app = QApplication(sys.argv)
    window = FindPersonApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
