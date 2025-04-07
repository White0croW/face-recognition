import sqlite3
import numpy as np


class FaceDatabase:
    def __init__(self, db_path="faces.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                embedding BLOB,
                image BLOB
            )
        """
        )

    def insert_embedding(self, embedding, image_bytes):
        self.conn.execute(
            "INSERT INTO faces (embedding, image) VALUES (?, ?)",
            (np.array(embedding).tobytes(), image_bytes),
        )
        self.conn.commit()

    def find_similar(self, query_embedding, threshold=0.6):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM faces")
        matches = []
        for row in cursor.fetchall():
            db_embedding = np.frombuffer(row[1], dtype=np.float32)
            similarity = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
            )
            if similarity > threshold:
                matches.append(
                    {"id": row[0], "image": row[2], "similarity": similarity}
                )
        return sorted(matches, key=lambda x: -x["similarity"])


def get_db_connection():
    return FaceDatabase()
