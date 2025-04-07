import numpy as np
from deepface import DeepFace
import cv2


class FaceService:
    def __init__(self, database):
        self.db = database

    def add_image_to_db(self, image_bytes: bytes):
        if not self._validate_image(image_bytes):
            raise ValueError("Некорректное изображение")

        # Детектирование лиц
        faces = DeepFace.extract_faces(
            img_path=image_bytes, detector_backend="mtcnn", enforce_detection=False
        )

        if not faces:
            raise ValueError("Лица не обнаружены")

        embeddings = []
        face_locations = []
        for face in faces:
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )
            face_roi = self._crop_face(image_bytes, y, x, y + h, x + w)
            embedding = self._extract_embedding(face_roi)
            if embedding is not None:
                embeddings.append(embedding)
                face_locations.append((y, x, y + h, x + w))  # top, left, bottom, right

        if not embeddings:
            raise ValueError("Не удалось извлечь эмбеддинги")

        self.db.save_image(image_bytes, embeddings, face_locations)

    def recognize_face(self, image_bytes: bytes, threshold=0.6):
        query_faces = self._process_query_image(image_bytes)
        if not query_faces:
            return []

        matches = []
        for db_face in self.db.get_all_faces():
            db_embeddings = db_face["embeddings"]
            db_locations = db_face["face_locations"]

            for q_emb, q_loc in query_faces:
                for db_emb, db_loc in zip(db_embeddings, db_locations):
                    similarity = self._calculate_similarity(q_emb, db_emb)
                    if similarity >= threshold:
                        matches.append(
                            {
                                "image": db_face["image"],
                                "face_location": db_loc,
                                "similarity": similarity,
                            }
                        )

        return sorted(matches, key=lambda x: -x["similarity"])

    def _validate_image(self, image_bytes: bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img is not None and img.size > 0
        except Exception:
            return False

    def _crop_face(self, image_bytes, top, left, bottom, right):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img[top:bottom, left:right]

    def _extract_embedding(self, face_roi):
        try:
            result = DeepFace.represent(
                img_path=face_roi, model_name="ArcFace", enforce_detection=False
            )
            return np.array(result[0]["embedding"], dtype=np.float32).tobytes()
        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга: {e}")
            return None

    def _calculate_similarity(self, emb1, emb2):
        vec1 = np.frombuffer(emb1, dtype=np.float32)
        vec2 = np.frombuffer(emb2, dtype=np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return np.dot(vec1, vec2)

    def _process_query_image(self, image_bytes):
        faces = DeepFace.extract_faces(
            img_path=image_bytes, detector_backend="mtcnn", enforce_detection=False
        )

        query_data = []
        for face in faces:
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )
            face_roi = self._crop_face(image_bytes, y, x, y + h, x + w)
            embedding = self._extract_embedding(face_roi)
            if embedding is not None:
                query_data.append((embedding, (y, x, y + h, x + w)))

        return query_data
