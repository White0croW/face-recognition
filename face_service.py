import numpy as np
import cv2
from deepface import DeepFace


class FaceService:
    def __init__(self, database):
        self.db = database

    def add_image_to_db(self, image_bytes: bytes):
        """Добавление изображения в БД после проверки"""
        if not self._validate_image(image_bytes):
            raise ValueError("Некорректное изображение")
        self.db.save_image(image_bytes)

    def recognize_face(self, image_bytes: bytes, threshold=0.6):
        """Распознавание лица и поиск совпадений"""
        query_embedding = self._extract_embedding(image_bytes)
        if query_embedding is None:
            return []

        matches = []
        for db_img in self.db.get_all_images():
            db_embedding = self._extract_embedding(db_img)
            if db_embedding is None:
                continue
            similarity = self._calculate_similarity(query_embedding, db_embedding)
            if similarity >= threshold:
                matches.append({"image": db_img, "similarity": similarity})
        return sorted(matches, key=lambda x: -x["similarity"])

    def _validate_image(self, image_bytes: bytes):
        """Валидация изображения через OpenCV"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img is not None and img.size > 0

    def _extract_embedding(self, image_bytes: bytes):
        """Извлечение эмбеддинга лица"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            result = DeepFace.represent(
                img, model_name="ArcFace", enforce_detection=False
            )
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            return embedding.tobytes()
        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга: {e}")
            return None

    def _calculate_similarity(self, emb1: bytes, emb2: bytes) -> float:
        """Вычисление косинусного сходства"""
        vec1 = np.frombuffer(emb1, dtype=np.float32)
        vec2 = np.frombuffer(emb2, dtype=np.float32)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
