import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceEncoder:
    """
    Encodes a reference image into a face embedding using InsightFace.
    """
    def __init__(self):
        # Load the InsightFace model
        self.app = FaceAnalysis(name="buffalo_l")  # large model for accuracy
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def encode(self, image_path: str) -> np.ndarray:
        """
        Reads an image file and returns the face embedding.
        Raises an error if no face is detected.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Reference image not found: {image_path}")

        faces = self.app.get(image)
        if not faces:
            raise ValueError(f"No face detected in reference image: {image_path}")

        # Return the first face embedding
        return faces[0].embedding
