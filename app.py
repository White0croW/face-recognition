import streamlit as st
from image_utils import process_image
from face_model import recognize_face
from database import get_db_connection

st.title("Распознавание лиц")

uploaded_file = st.file_uploader("Загрузите фото", type=["jpg", "png"])
if uploaded_file:
    image = process_image(uploaded_file)
    if image is not None:
        embedding = recognize_face(image)
        if embedding is not None:
            conn = get_db_connection()
            matches = conn.find_similar(embedding)
            st.write(f"Найдено совпадений: {len(matches)}")
            for match in matches:
                st.image(match["image"], caption=f"Сходство: {match['similarity']:.2f}")
        else:
            st.error("Лицо не обнаружено")
