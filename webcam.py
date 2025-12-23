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
        found = False
        best_score = 0.0

        faces = self.app.get(frame)
        for face in faces:
            score = cosine_similarity(face.embedding, self.reference_embedding)
            best_score = max(best_score, score)

            if score >= self.threshold:
                found = True
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"MATCH {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        return frame, found, best_score
