import cv2


class VideoFeed:
    def __init__(self, finder):
        self.finder = finder
        self.cap = cv2.VideoCapture(finder.video_source)
        self.running = False

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video source")

    def start(self):
        self.running = True
        window_name = "Find Person"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # -------------------------------------------------
        # VIDEO WARM-UP (IMPORTANT FOR RECORDED VIDEOS)
        # -------------------------------------------------
        # Recorded videos often have unstable first frames.
        # Skipping them prevents tracking glitches.
        is_video_file = isinstance(self.finder.video_source, str)
        warmup_frames = 10 if is_video_file else 0

        for _ in range(warmup_frames):
            ret, _ = self.cap.read()
            if not ret:
                break
        # -------------------------------------------------

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, found, score = self.finder.detect_frame(frame)

            if found:
                text = f"FOUND | Score: {score:.2f}"
                color = (0, 255, 0)
            else:
                text = f"SEARCHING | Best: {score:.2f}"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

            cv2.imshow(window_name, frame)

            # ESC fallback
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self._cleanup()

    def stop(self):
        # Non-blocking stop (GUI-safe)
        self.running = False

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
