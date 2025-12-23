import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QSlider,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPixmap

from face_encoder import FaceEncoder
from webcam import VideoFinder
from backend.video_feed import VideoFeed


# ===================== Worker Thread =====================

class VideoWorker(QThread):
    def __init__(self, ref_images, source, threshold):
        super().__init__()
        self.ref_images = ref_images
        self.source = source
        self.threshold = threshold
        self.feed = None

    def run(self):
        encoder = FaceEncoder()
        reference_embeddings = encoder.encode_images(self.ref_images)

        finder = VideoFinder(
            reference_embeddings=reference_embeddings,
            video_source=self.source,
            threshold=self.threshold,
        )

        self.feed = VideoFeed(finder)
        self.feed.start()

    def stop(self):
        if self.feed:
            self.feed.stop()
        self.quit()
        self.wait(100)


# ===================== Main GUI =====================

class FindPersonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Find Person System")
        self.setFixedSize(500, 550)

        self.ref_images = []        # ALWAYS a list
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
        self.image_preview = QLabel("No reference images")
        self.image_preview.setFixedSize(200, 200)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_preview, alignment=Qt.AlignCenter)

        # Upload button
        btn_ref = QPushButton("Add Reference Image")
        btn_ref.clicked.connect(self.add_reference_image)
        layout.addWidget(btn_ref)

        # Upload info label
        self.info_label = QLabel("No images uploaded")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

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

        # Controls
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
        btn_stop.setStyleSheet(
            "background-color: red; color: white; font-weight: bold;"
        )
        btn_stop.clicked.connect(self.stop_feed)
        layout.addWidget(btn_stop)

        self.setLayout(layout)

    # ===================== Actions =====================

    def add_reference_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            "",
            "Images (*.jpg *.jpeg *.png)",
        )

        if not path:
            return

        # Prevent duplicates
        if path in self.ref_images:
            QMessageBox.information(
                self, "Info", "This image is already added."
            )
            return

        self.ref_images.append(path)

        # Update preview (show last added image)
        pix = QPixmap(path).scaled(
            200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_preview.setPixmap(pix)

        # Update info message
        count = len(self.ref_images)
        self.info_label.setText(f"You have uploaded {count} image(s).")

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi *.mov)",
        )
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
        if not self.ref_images:
            QMessageBox.warning(
                self, "Error", "Upload at least one reference image."
            )
            return

        if self.worker:
            QMessageBox.warning(self, "Error", "Video already running.")
            return

        self.worker = VideoWorker(
            self.ref_images,
            source,
            self.slider.value() / 100,
        )
        self.worker.start()

    def stop_feed(self):
        if self.worker:
            self.worker.stop()
            self.worker = None


# ===================== App Entry =====================

def main():
    app = QApplication(sys.argv)
    window = FindPersonApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
