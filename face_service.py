import numpy as np
import cv2
from deepface import DeepFace
from database import SQLiteDB


class FaceService:
    def __init__(self, database: SQLiteDB):
        self.db = database

    def add_image_to_db(self, image_bytes: bytes):
        if not image_bytes:
            return

        # Проверка изображения через OpenCV [[5]]
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("Некорректный формат изображения")

        self.db.save_image(image_bytes)

    def recognize_face(self, image_bytes: bytes, threshold=0.6):
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

    def get_all_images(self):
        return self.db.get_all_images()

    def _extract_embedding(self, image_bytes: bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            result = DeepFace.represent(
                img, model_name="ArcFace", enforce_detection=False
            )
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            return embedding.tobytes()
        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга: {e}")
            return None

    def _calculate_similarity(self, emb1: bytes, emb2: bytes) -> float:
        vec1 = np.frombuffer(emb1, dtype=np.float32)
        vec2 = np.frombuffer(emb2, dtype=np.float32)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
