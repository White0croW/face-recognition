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
                embedding BLOB NOT NULL,
                image BLOB NOT NULL
            )
        """
        )

    def save_face(self, embedding: bytes, image: bytes):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO faces (embedding, image) VALUES (?, ?)", (embedding, image)
        )
        self.conn.commit()

    def find_matches(self, query_embedding: bytes, threshold=0.6):
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding, image FROM faces")

        matches = []
        query = np.frombuffer(query_embedding, dtype=np.float32)

        for row in cursor.fetchall():
            db_embedding = np.frombuffer(row[0], dtype=np.float32)
            similarity = np.dot(query, db_embedding) / (
                np.linalg.norm(query) * np.linalg.norm(db_embedding)
            )

            if similarity >= threshold:
                matches.append({"similarity": similarity, "image": row[1]})

        return sorted(matches, key=lambda x: -x["similarity"])
