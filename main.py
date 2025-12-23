# def main():
#     print("Hello from find-person!")


# if __name__ == "__main__":
#     main()


from face_encoder import FaceEncoder
from webcam import VideoFinder

def main():
    # Encode reference image
    encoder = FaceEncoder()
    reference_embedding = encoder.encode("data/reference.jpg")

    print("Reference face encoded successfully!")

    # Ask user for video source
    video_source = input("Enter video file path or leave blank for webcam: ").strip()
    video_source = 0 if video_source == "" else video_source

    # Initialize VideoFinder with the chosen source
    finder = VideoFinder(reference_embedding, video_source=video_source)

    print("Starting detection... Press Q to quit.")
    finder.run()

if __name__ == "__main__":
    main()
