import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread

from face_encoder import FaceEncoder
from webcam import VideoFinder
from backend.video_feed import VideoFeed


class VideoWorker(QThread):
    """
    Runs video processing in a separate thread
    """

    def __init__(self, reference_image_path, video_source):
        super().__init__()
        self.reference_image_path = reference_image_path
        self.video_source = video_source

    def run(self):
        # Encode reference image
        encoder = FaceEncoder()
        reference_embedding = encoder.encode(self.reference_image_path)

        # Create video finder
        finder = VideoFinder(
            reference_embedding=reference_embedding,
            video_source=self.video_source,
        )

        # Start video feed
        feed = VideoFeed(finder)
        feed.start()


class FindPersonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Find Person System")
        self.setFixedSize(420, 320)

        self.reference_image_path = None
        self.video_path = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Person Finder")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        self.ref_label = QLabel("No reference image selected")
        self.ref_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.ref_label)

        btn_ref = QPushButton("Upload Reference Image")
        btn_ref.clicked.connect(self.upload_reference_image)
        layout.addWidget(btn_ref)

        layout.addSpacing(10)

        btn_live = QPushButton("Start Live Webcam")
        btn_live.clicked.connect(self.start_live_feed)
        layout.addWidget(btn_live)

        layout.addSpacing(10)

        btn_video = QPushButton("Upload Recorded Video")
        btn_video.clicked.connect(self.upload_video)
        layout.addWidget(btn_video)

        btn_start_video = QPushButton("Start Recorded Video")
        btn_start_video.clicked.connect(self.start_recorded_video)
        layout.addWidget(btn_start_video)

        self.setLayout(layout)

    # ---------------- UI Actions ---------------- #

    def upload_reference_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            "",
            "Images (*.jpg *.jpeg *.png)",
        )
        if path:
            self.reference_image_path = path
            self.ref_label.setText(path.split("/")[-1])

    def upload_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Videos (*.mp4 *.avi *.mov)",
        )
        if path:
            self.video_path = path

    def start_live_feed(self):
        if not self.reference_image_path:
            QMessageBox.critical(
                self, "Error", "Please upload a reference image first."
            )
            return

        self.worker = VideoWorker(
            reference_image_path=self.reference_image_path,
            video_source=0,  # webcam
        )
        self.worker.start()

    def start_recorded_video(self):
        if not self.reference_image_path or not self.video_path:
            QMessageBox.critical(
                self,
                "Error",
                "Please upload both reference image and video file.",
            )
            return

        self.worker = VideoWorker(
            reference_image_path=self.reference_image_path,
            video_source=self.video_path,
        )
        self.worker.start()


def main():
    app = QApplication(sys.argv)
    window = FindPersonApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
