import numpy as np
from deepface import DeepFace
from database import SQLiteDB
import cv2


class FaceService:

    def __init__(self, database: SQLiteDB):
        self.db = database

    def add_image_to_db(self, image_bytes: bytes):
        """Сохраняет изображение в БД без распознавания лица"""
        self.db.save_image(image_bytes)  # Только сохранение, без анализа

    def recognize_face(self, image_bytes: bytes, threshold=0.6):
        """Распознает лицо и ищет совпадения в БД"""
        query_embedding = self._extract_embedding(image_bytes)
        if query_embedding is None:
            return []

        # Получаем все изображения из БД и анализируем их на лету
        images = self.db.get_all_images()
        matches = []
        for img in images:
            db_embedding = self._extract_embedding(img)
            if db_embedding is None:
                continue  # Пропускаем изображения без лиц
            similarity = self._calculate_similarity(query_embedding, db_embedding)
            if similarity >= threshold:
                matches.append({"image": img, "similarity": similarity})
        return sorted(matches, key=lambda x: -x["similarity"])
