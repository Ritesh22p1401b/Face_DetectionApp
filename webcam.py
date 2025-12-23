import cv2
from insightface.app import FaceAnalysis
from face_matcher import cosine_similarity


# ===================== Tracker Factory =====================

def create_tracker():
    """
    Create the best available OpenCV tracker.
    Works with opencv-contrib-python and different OpenCV versions.
    """
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()

    # OpenCV 4.5+ legacy namespace fallback
    if hasattr(cv2, "legacy"):
        if hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
        if hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()

    raise RuntimeError(
        "No OpenCV tracker available. Install opencv-contrib-python."
    )


# ===================== Video Finder =====================

class VideoFinder:
    def __init__(
        self,
        reference_embeddings,
        video_source=0,
        threshold=0.5,
        detect_interval=5,
    ):
        self.reference_embeddings = reference_embeddings
        self.video_source = video_source
        self.threshold = threshold
        self.detect_interval = detect_interval

        # InsightFace (CPU / AMD-safe)
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        self.frame_count = 0
        self.tracker = None
        self.tracking = False
        self.last_score = 0.0

    def _best_similarity(self, embedding):
        return max(
            cosine_similarity(embedding, ref)
            for ref in self.reference_embeddings
        )

    def detect_frame(self, frame):
        self.frame_count += 1
        found = False
        best_score = 0.0

        # ---------- Tracking path (FAST) ----------
        if self.tracking and self.tracker:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                return frame, True, self.last_score
            else:
                # Tracker lost target → fallback to detection
                self.tracking = False
                self.tracker = None

        # ---------- Detection path (SLOW) ----------
        if self.frame_count % self.detect_interval == 0:
            faces = self.app.get(frame)

            for face in faces:
                score = self._best_similarity(face.embedding)
                best_score = max(best_score, score)

                if score >= self.threshold:
                    found = True
                    x1, y1, x2, y2 = map(int, face.bbox)

                    # Initialize tracker safely
                    self.tracker = create_tracker()
                    self.tracker.init(
                        frame,
                        (x1, y1, x2 - x1, y2 - y1),
                    )

                    self.tracking = True
                    self.last_score = score

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"MATCH {score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    break

        return frame, found, best_score
