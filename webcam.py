import cv2
from insightface.app import FaceAnalysis
from face_matcher import cosine_similarity

class VideoFinder:
    def __init__(self, reference_embedding, video_source=0, threshold=0.5):
        self.reference_embedding = reference_embedding
        self.video_source = video_source
        self.threshold = threshold

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_frame(self, frame):
        """Detect faces in a single frame and draw boxes if matched."""
        faces = self.app.get(frame)
        for face in faces:
            score = cosine_similarity(face.embedding, self.reference_embedding)
            if score > self.threshold:
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"FOUND ({score:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source {self.video_source}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detect_frame(frame)
            cv2.imshow("Find Person", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
