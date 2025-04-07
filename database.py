import sqlite3


class SQLiteDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                image BLOB NOT NULL
            )
        """
        )

    def save_image(self, image_bytes: bytes):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO faces (image) VALUES (?)", (image_bytes,))
        self.conn.commit()

    def get_all_images(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT image FROM faces")
        return [row[0] for row in cursor.fetchall()]

    def get_image_count(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faces")
        return cursor.fetchone()[0]
