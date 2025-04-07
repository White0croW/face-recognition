import sqlite3
import numpy as np


class SQLiteDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                image BLOB NOT NULL  -- Убрано поле embedding
            )
        """
        )

    def save_image(self, image_bytes: bytes):
        """Сохраняет изображение без анализа"""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO faces (image) VALUES (?)", (image_bytes,))
        self.conn.commit()

    def get_all_images(self):
        """Возвращает все изображения из БД"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT image FROM faces")
        return [row[0] for row in cursor.fetchall()]
