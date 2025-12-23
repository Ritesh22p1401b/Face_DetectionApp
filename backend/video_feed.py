import cv2
from webcam import VideoFinder

class VideoStream:
    def __init__(self, video_finder: VideoFinder):
        """
        video_finder: Instance of VideoFinder with reference embedding loaded
        """
        self.video_finder = video_finder
        self.cap = cv2.VideoCapture(self.video_finder.video_source)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video source")

    def generate_frames(self):
        """
        Generator function that yields MJPEG frames for FastAPI StreamingResponse
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Run face detection and draw boxes
            frame = self.video_finder.detect_frame(frame)

            # Encode frame as JPEG
            success, jpeg = cv2.imencode('.jpg', frame)
            if not success:
                continue

            # Yield in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    def release(self):
        self.cap.release()
