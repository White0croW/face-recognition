import cv2
import numpy as np
from PIL import Image


def process_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_image(image: np.ndarray) -> bool:
    return image is not None and image.size > 0
