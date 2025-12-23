# import cv2


# class VideoFeed:
#     def __init__(self, finder):
#         self.finder = finder
#         self.cap = cv2.VideoCapture(finder.video_source)
#         self.running = True

#         if not self.cap.isOpened():
#             raise RuntimeError("Cannot open video source")

#     def start(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             frame, found, score = self.finder.detect_frame(frame)

#             status = "FOUND" if found else "NOT FOUND"
#             color = (0, 255, 0) if found else (0, 0, 255)

#             cv2.putText(
#                 frame,
#                 f"{status} | Confidence: {score:.2f}",
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 color,
#                 2,
#             )

#             cv2.imshow("Find Person", frame)

#             if cv2.waitKey(1) & 0xFF == 27:  # ESC fallback
#                 break

#         self.stop()

#     def stop(self):
#         self.running = False
#         self.cap.release()
#         cv2.destroyAllWindows()


# reframe speed


# import cv2
# import time


# class VideoFeed:
#     def __init__(self, finder):
#         self.finder = finder
#         self.cap = cv2.VideoCapture(finder.video_source)
#         self.running = True

#         if not self.cap.isOpened():
#             raise RuntimeError("Cannot open video source")

#         # Detect if source is recorded video
#         self.is_video_file = isinstance(finder.video_source, str)

#         # Get FPS for recorded video
#         self.fps = self.cap.get(cv2.CAP_PROP_FPS)
#         if self.fps <= 0:
#             self.fps = 25  # fallback

#         self.frame_delay = 1.0 / self.fps

#     def start(self):
#         window_name = "Find Person"

#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#         # Fullscreen ONLY for recorded video
#         if self.is_video_file:
#             cv2.setWindowProperty(
#                 window_name,
#                 cv2.WND_PROP_FULLSCREEN,
#                 cv2.WINDOW_FULLSCREEN
#             )

#         frame_count = 0
#         skip_rate = 2 if self.is_video_file else 1  # skip frames for video

#         while self.running:
#             start_time = time.time()

#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             frame_count += 1
#             if frame_count % skip_rate != 0:
#                 continue

#             # Resize frame for speed (VERY IMPORTANT)
#             frame = cv2.resize(frame, (960, 540))

#             frame, found, score = self.finder.detect_frame(frame)

#             status = "FOUND" if found else "NOT FOUND"
#             color = (0, 255, 0) if found else (0, 0, 255)

#             cv2.putText(
#                 frame,
#                 f"{status} | Confidence: {score:.2f}",
#                 (30, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 color,
#                 2,
#             )

#             cv2.imshow(window_name, frame)

#             elapsed = time.time() - start_time
#             delay = max(1, int((self.frame_delay - elapsed) * 1000))

#             if cv2.waitKey(delay) & 0xFF == 27:  # ESC
#                 break

#         self.stop()

#     def stop(self):
#         self.running = False
#         self.cap.release()
#         cv2.destroyAllWindows()



import cv2
import time


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

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, found, score = self.finder.detect_frame(frame)

            cv2.putText(
                frame,
                f"{'FOUND' if found else 'NOT FOUND'} | {score:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )

            cv2.imshow(window_name, frame)

            # IMPORTANT: short waitKey so loop can exit fast
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self._cleanup()

    def stop(self):
        # Just flip the flag — DO NOT block
        self.running = False

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
