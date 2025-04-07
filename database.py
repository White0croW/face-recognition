import sqlite3
import numpy as np


class SQLiteDB:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Создание таблиц в базе данных"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                image BLOB NOT NULL
            )
        """
        )
        self.conn.commit()

    def save_image(self, image_bytes: bytes):
        """Сохранение изображения в базу данных"""
        cursor = self.conn.cursor()
        embedding = self._extract_embedding(image_bytes)
        if embedding is None:
            raise ValueError("Не удалось извлечь эмбеддинг из изображения")
        cursor.execute(
            """
            INSERT INTO faces (embedding, image) VALUES (?, ?)
        """,
            (embedding, image_bytes),
        )
        self.conn.commit()

    def get_all_images(self):
        """Получение всех изображений из базы данных"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT image FROM faces")
        return [row[0] for row in cursor.fetchall()]

    def get_image_count(self):
        """Получение количества изображений в базе данных"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faces")
        return cursor.fetchone()[0]

    def _extract_embedding(self, image_bytes: bytes):
        """Извлечение эмбеддинга лица из изображения"""
        try:
            from deepface import DeepFace

            embedding = DeepFace.represent(
                image_bytes, model_name="ArcFace", enforce_detection=False
            )[0]["embedding"]
            return np.array(embedding, dtype=np.float32).tobytes()
        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга: {e}")
            return None
