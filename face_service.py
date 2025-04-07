import numpy as np
import cv2
from deepface import DeepFace


class FaceService:
    def __init__(self, database):
        self.db = database

    def add_image_to_db(self, image_bytes: bytes):
        if not self._validate_image(image_bytes):
            raise ValueError("Некорректное изображение")

        # Преобразуем байты в массив
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = np.ascontiguousarray(img)  # Для совместимости с OpenCV

        faces = DeepFace.extract_faces(
            img_path=img,  # Передаем массив, а не байты
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
            face_roi = img[y : y + h, x : x + w]  # Обрезка из массива
            embedding = self._extract_embedding(face_roi)
            if embedding is not None:
                embeddings.append(embedding.tobytes())  # tobytes() для массива
                face_locations.append((y, x, y + h, x + w))

        if not embeddings:
            raise ValueError("Не удалось извлечь эмбеддинги")

        self.db.save_image(image_bytes, embeddings, face_locations)

    def _extract_embedding(self, face_roi):
        try:
            # Убедимся, что face_roi — массив NumPy
            if isinstance(face_roi, bytes):
                nparr = np.frombuffer(face_roi, np.uint8)
                face_roi = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            face_roi = np.ascontiguousarray(
                face_roi
            )  # Исправление для C-contiguous [[4]]

            result = DeepFace.represent(
                img_path=face_roi, model_name="ArcFace", enforce_detection=False
            )
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            return embedding  # Возвращаем массив, а не байты
        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга: {e}")
            return None

    def _crop_face(self, img, top, left, bottom, right):
        face_roi = img[top:bottom, left:right]
        return np.ascontiguousarray(face_roi)  # Добавлено [[4]]

    def recognize_face(self, image_bytes: bytes, threshold=0.6):
        query_faces = self._process_query_image(image_bytes)
        if not query_faces:
            return []

        matches = []
        for db_face in self.db.get_all_faces():
            for q_emb, q_loc in query_faces:
                for db_emb, db_loc in zip(
                    db_face["embeddings"], db_face["face_locations"]
                ):
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
