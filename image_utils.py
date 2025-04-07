import cv2
import numpy as np
from PIL import Image


def process_image(file):
    """Преобразует файл в массив OpenCV."""
    img = Image.open(file)
    img_array = np.array(img)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
