# image_utils.py (улучшенная версия)
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import hashlib


def generate_filename(prefix="img"):
    """Генерация уникального имени файла"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"


def calculate_md5(image_bytes):
    """Вычисление хеша MD5 для изображения"""
    return hashlib.md5(image_bytes).hexdigest()


def process_image(file_bytes: bytes) -> np.ndarray:
    """Улучшенная обработка изображения с проверками"""
    try:
        if not isinstance(file_bytes, bytes):
            raise ValueError("Input must be bytes")

        if len(file_bytes) == 0:
            raise ValueError("Empty image data")

        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Неверный формат изображения (поддерживаются JPG/PNG)")

        # Конвертация и нормализация
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Автоповорот по EXIF (для JPEG)
        pil_img = Image.fromarray(image)
        if hasattr(pil_img, '_getexif'):
            exif = pil_img._getexif()
            if exif:
                orientation = exif.get(0x0112)
                # Логика поворота...

        return np.array(pil_img)

    except Exception as e:
        print(f"[Ошибка] {str(e)}")
        return None


def validate_image(image: np.ndarray, min_size=32) -> bool:
    """Расширенная валидация изображения"""
    if image is None:
        return False

    if not isinstance(image, np.ndarray):
        return False

    # Проверка размеров
    if image.ndim not in [2, 3]:
        return False

    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
        return False

    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        return False

    # Проверка содержания (не все нули)
    if np.all(image == 0):
        return False

    return True


def handle_upload(file_bytes: bytes, save_dir: str = None) -> str:
    """Обработка загрузки с сохранением"""
    try:
        img_array = process_image(file_bytes)
        if not validate_image(img_array):
            return "Невалидное изображение"

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = generate_filename()
            save_path = os.path.join(save_dir, filename)
            Image.fromarray(img_array).save(save_path)
            return f"Сохранено: {filename}"
        return "Обработано успешно"
    except Exception as e:
        return f"Ошибка: {str(e)}"


def handle_search(query: str, search_dir: str) -> list:
    """Улучшенный поиск изображений"""
    if not os.path.exists(search_dir):
        return ["Директория не существует"]

    results = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if query.lower() in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                results.append(os.path.join(root, file))

    return results if results else ["Ничего не найдено"]