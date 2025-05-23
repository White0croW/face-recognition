import sqlite3
import pickle
import numpy as np
import cv2


class SQLiteDB:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embeddings BLOB NOT NULL,
                face_locations BLOB NOT NULL,
                image BLOB NOT NULL
            )
        """
        )
        self.conn.commit()

    def save_image(self, image: np.ndarray, embeddings, face_locations):
        # Конвертируем массив в байты перед сохранением [[5]]
        _, buffer = cv2.imencode(".jpg", image)
        image_bytes = buffer.tobytes()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO faces (embeddings, face_locations, image)
            VALUES (?, ?, ?)
        """,
            (
                pickle.dumps(embeddings),
                pickle.dumps(face_locations),
                image_bytes,  # Сохраняем байты, а не массив [[5]]
            ),
        )
        self.conn.commit()

    def get_all_faces(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embeddings, face_locations, image FROM faces")
        return [
            {
                "id": row[0],
                "embeddings": pickle.loads(row[1]),
                "face_locations": pickle.loads(row[2]),
                "image": row[3],
            }
            for row in cursor.fetchall()
        ]

    def delete_face(self, face_id: int):
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM faces WHERE id = ?",
            (face_id,)
        )
        self.conn.commit()
        return cursor.rowcount  # Возвращает количество удаленных строк (1 если удалено, 0 если нет такой записи)