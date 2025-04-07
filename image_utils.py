import cv2
import numpy as np
from PIL import Image


def process_image(file_bytes: bytes) -> np.ndarray:
    try:
        # Преобразование байтов в массив NumPy [[1]][[3]]
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Проверка успешного декодирования [[1]][[3]]
        if image is None:
            raise ValueError(
                "Невозможно декодировать изображение. Проверьте формат (JPG/PNG)"
            )

        # Конвертация в RGB (OpenCV использует BGR по умолчанию) [[7]]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Ошибка обработки изображения: {e}")
        return None


def validate_image(image: np.ndarray) -> bool:
    # Проверка на None и наличие данных [[2]][[4]]
    if image is None:
        return False

    # Проверка, что изображение имеет 3 канала (цветное) [[2]][[4]]
    if image.ndim != 3 or image.shape[2] != 3:
        return False

    return image.size > 0
