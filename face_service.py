import numpy as np
import cv2
from deepface import DeepFace
import pickle


class FaceService:
    def __init__(self, database):
        self.db = database

    def add_image_to_db(self, image_bytes: bytes):
        if not self._validate_image(image_bytes):
            raise ValueError("Некорректное изображение")

        # Преобразование байтов в массив
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Детектирование лиц
        faces = DeepFace.extract_faces(
            img_path=img,  # Теперь передаем массив, а не байты
            detector_backend="mtcnn",
            enforce_detection=False,
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
            face_roi = img[y : y + h, x : x + w]  # Обрезка лица
            embedding = self._extract_embedding(face_roi)
            if embedding is not None:
                embeddings.append(embedding.tobytes())
                face_locations.append((y, x, y + h, x + w))

        if not embeddings:
            raise ValueError("Не удалось извлечь эмбеддинги")

        self.db.save_image(image_bytes, embeddings, face_locations)

    def recognize_face(self, image_bytes: bytes, threshold=0.6):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        query_faces = DeepFace.extract_faces(
            img_path=img, detector_backend="mtcnn", enforce_detection=False
        )

        matches = []
        for face in query_faces:
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )
            face_roi = img[y : y + h, x : x + w]
            query_embedding = self._extract_embedding(face_roi)

            for db_face in self.db.get_all_faces():
                for db_emb, db_loc in zip(
                    db_face["embeddings"], db_face["face_locations"]
                ):
                    similarity = self._calculate_similarity(query_embedding, db_emb)
                    if similarity >= threshold:
                        matches.append(
                            {
                                "image": db_face["image"],
                                "face_location": db_loc,
                                "similarity": similarity,
                            }
                        )

        return sorted(matches, key=lambda x: -x["similarity"])

    def _extract_embedding(self, face_roi):
        try:
            result = DeepFace.represent(
                img_path=face_roi, model_name="ArcFace", enforce_detection=False
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
