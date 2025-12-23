# import cv2
# from webcam import VideoFinder


# class VideoStream:
#     def __init__(self, video_finder: VideoFinder):
#         """
#         video_finder: Instance of VideoFinder with reference embedding loaded
#         """
#         self.video_finder = video_finder
#         self.cap = cv2.VideoCapture(self.video_finder.video_source)

#         if not self.cap.isOpened():
#             raise RuntimeError("Cannot open video source")

#         self.running = True

#     def start(self):
#         """
#         Start video processing loop (GUI / desktop usage)
#         """
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             # Run face detection and matching
#             frame = self.video_finder.detect_frame(frame)

#             # Show frame
#             cv2.imshow("Find Person - Video Feed", frame)

#             # Press 'q' to quit
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         self.release()

#     def stop(self):
#         self.running = False

#     def release(self):
#         self.cap.release()
#         cv2.destroyAllWindows()



import cv2
from webcam import VideoFinder


class VideoFeed:
    def __init__(self, video_finder: VideoFinder):
        self.video_finder = video_finder
        self.cap = cv2.VideoCapture(self.video_finder.video_source)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video source")

        self.running = True

    def start(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.video_finder.detect_frame(frame)

            cv2.imshow("Find Person - Video Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.release()

    def stop(self):
        self.running = False

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
