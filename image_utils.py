import os
from zipfile import ZipFile
from PIL import Image
import sqlite3


from face_model import FaceRecognizer  # Импорт модели


def process_recognition(image_path):
    """Распознавание лиц на изображении [[6]]"""
    recognizer = FaceRecognizer()
    image = Image.open(image_path)
    faces = recognizer.detect_faces(image)
    results = []
    for face in faces:
        embedding = recognizer.recognize(face)
        results.append({"coordinates": face, "embedding": embedding})
    return results


def process_single_image(file):
    """Сохранение отдельного изображения [[1]][[6]]"""
    filename = os.path.basename(file.name)
    save_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(file.read())
    return filename, save_path


def process_zip_archive(zip_file):
    """Обработка ZIP-архива [[7]]"""
    with ZipFile(zip_file) as zip:
        for filename in zip.namelist():
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                save_path = os.path.join("uploads", filename)
                with open(save_path, "wb") as f:
                    f.write(zip.read(filename))
                yield filename, save_path


def search_images(query):
    """Поиск изображений в БД [[5]]"""
    conn = sqlite3.connect("images.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_path FROM images WHERE filename LIKE ?", (f"%{query}%",)
    )
    return [row[0] for row in cursor.fetchall()]
