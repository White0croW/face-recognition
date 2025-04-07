import numpy as np
from deepface import DeepFace
from database import Database


class FaceService:
    def __init__(self, database: Database):
        self.db = database

    def add_image_to_db(self, image_bytes: bytes):
        embedding = self._extract_embedding(image_bytes)
        if embedding is not None:
            self.db.save_face(embedding, image_bytes)
        else:
            raise ValueError("Лицо не обнаружено")

    def recognize_face(self, image_bytes: bytes, threshold=0.6):
        query_embedding = self._extract_embedding(image_bytes)
        if query_embedding is None:
            return []
        return self.db.find_matches(query_embedding, threshold)

    def _extract_embedding(self, image_bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            result = DeepFace.represent(
                img, model_name="ArcFace", enforce_detection=False
            )
            return np.array(result[0]["embedding"]).tobytes()
        except Exception:
            return None
