from deepface import DeepFace
import numpy as np


def recognize_face(image):
    """Извлекает эмбеддинг лица из изображения."""
    try:
        embedding = DeepFace.represent(
            image, model_name="ArcFace", enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Ошибка распознавания: {e}")
        return None


def find_similar(self, query_embedding, threshold=0.6):
    """Поиск похожих лиц с использованием косинусного сходства [[4]][[6]]"""
    cursor = self.conn.cursor()
    cursor.execute("SELECT * FROM faces")
    matches = []
    query = np.frombuffer(query_embedding, dtype=np.float32)
    for row in cursor.fetchall():
        db_embedding = np.frombuffer(row[1], dtype=np.float32)
        similarity = np.dot(query, db_embedding) / (
            np.linalg.norm(query) * np.linalg.norm(db_embedding)
        )
        if similarity > threshold:
            matches.append({"id": row[0], "image": row[2], "similarity": similarity})
    return sorted(matches, key=lambda x: -x["similarity"])
