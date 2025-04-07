import os
import streamlit as st
from database import SQLiteDB
from face_service import FaceService
from ui import FaceRecognitionUI

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Отключаем предупреждения TensorFlow


def main():
    db = SQLiteDB("faces.db")
    service = FaceService(db)
    ui = FaceRecognitionUI(service)
    ui.render()


if __name__ == "__main__":
    main()
