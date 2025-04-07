import streamlit as st
from ui import FaceRecognitionUI
from database import SQLiteDB
from face_service import FaceService


def main():
    db = SQLiteDB("faces.db")
    service = FaceService(db)
    ui = FaceRecognitionUI(service)
    ui.render()


if __name__ == "__main__":
    main()
