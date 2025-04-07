import numpy as np
import cv2
from deepface import DeepFace


class FaceService:
    def __init__(self, database):
        self.db = database

    def add_image_to_db(self, image_bytes: bytes):
        if not self._validate_image(image_bytes):
            raise ValueError("Некорректное изображение")

        # Детектирование лиц и извлечение эмбеддингов
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Ошибка декодирования изображения")

        # Обнаружение лиц с помощью MTCNN
        detected_faces = DeepFace.extract_faces(
            img, detector_backend="mtcnn", enforce_detection=False
        )
        if not detected_faces:
            raise ValueError("На изображении не обнаружено лиц")

        embeddings = []
        face_locations = []
        for face in detected_faces:
            facial_area = face["facial_area"]
            x, y, w, h = (
                facial_area["x"],
                facial_area["y"],
                facial_area["w"],
                facial_area["h"],
            )
            top, right, bottom, left = y, x + w, y + h, x
            face_roi = img[top:bottom, left:right]

            # Извлечение эмбеддинга
            embedding = self._extract_embedding(face_roi)
            if embedding is not None:
                embeddings.append(embedding.tobytes())
                face_locations.append((top, right, bottom, left))

        if not embeddings:
            raise ValueError("Не удалось извлечь эмбеддинги")

        self.db.save_image(image_bytes, embeddings, face_locations)

    def recognize_face(self, image_bytes: bytes, threshold=0.6):
        query_faces = self._extract_query_faces(image_bytes)
        if not query_faces:
            return []

        matches = []
        for query_face in query_faces:
            query_embedding = query_face["embedding"]
            query_location = query_face["location"]

            for db_face in self.db.get_all_faces():
                db_image = db_face["image"]
                db_embeddings = db_face["embeddings"]
                db_locations = db_face["face_locations"]

                for db_emb, db_loc in zip(db_embeddings, db_locations):
                    similarity = self._calculate_similarity(query_embedding, db_emb)
                    if similarity >= threshold:
                        matches.append(
                            {
                                "image": db_image,
                                "face_location": db_loc,
                                "similarity": similarity,
                            }
                        )

        return sorted(matches, key=lambda x: -x["similarity"])

    def _extract_query_faces(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        detected_faces = DeepFace.extract_faces(
            img, detector_backend="mtcnn", enforce_detection=False
        )
        query_faces = []
        for face in detected_faces:
            facial_area = face["facial_area"]
            x, y, w, h = (
                facial_area["x"],
                facial_area["y"],
                facial_area["w"],
                facial_area["h"],
            )
            top, right, bottom, left = y, x + w, y + h, x
            face_roi = img[top:bottom, left:right]

            embedding = self._extract_embedding(face_roi)
            if embedding is not None:
                query_faces.append(
                    {"embedding": embedding, "location": (top, right, bottom, left)}
                )

        return query_faces

    def _extract_embedding(self, face_roi):
        try:
            result = DeepFace.represent(
                face_roi, model_name="ArcFace", enforce_detection=False
            )
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            return embedding.tobytes()
        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга: {e}")
            return None

    def _calculate_similarity(self, emb1, emb2):
        vec1 = np.frombuffer(emb1, dtype=np.float32)
        vec2 = np.frombuffer(emb2, dtype=np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return np.dot(vec1, vec2)
