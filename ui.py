import streamlit as st
from zipfile import ZipFile
from io import BytesIO


class FaceRecognitionUI:
    def __init__(self, service):
        self.service = service

    def render(self):
        st.title("Система распознавания лиц")
        menu = st.sidebar.selectbox("Меню", ["Добавить в БД", "Распознать"])

        if menu == "Добавить в БД":
            self._render_db_upload()
        else:
            self._render_recognition()

    def _render_db_upload(self):
        st.header("Добавление данных")
        files = st.file_uploader(
            "Выберите файлы/архивы",
            type=["jpg", "png", "zip"],
            accept_multiple_files=True,
        )

        if files:
            total_files = self._count_files(files)
            progress_bar = st.progress(0)
            success = 0

            for i, file in enumerate(files):
                if file.type == "application/zip":
                    with ZipFile(file) as z:
                        filelist = [
                            f
                            for f in z.namelist()
                            if f.lower().endswith((".jpg", ".png"))
                        ]
                        for j, name in enumerate(filelist):
                            self.service.add_image_to_db(z.read(name))
                            progress_bar.progress((i + j / len(filelist)) / total_files)
                else:
                    self.service.add_image_to_db(file.getvalue())
                    progress_bar.progress((i + 1) / total_files)
                success += 1

            progress_bar.empty()
            st.success(f"✅ Загружено файлов: {success}")

    def _process_zip(self, zip_file):
        with ZipFile(zip_file) as z:
            for name in z.namelist():
                if name.lower().endswith((".jpg", ".png")):
                    self.service.add_image_to_db(z.read(name))

    def _render_recognition(self):
        st.header("Распознавание")
        col1, col2 = st.columns(2)

        with col1:
            uploaded = st.file_uploader("Загрузить фото", type=["jpg", "png"])
        with col2:
            camera = st.camera_input("Использовать камеру")

        if uploaded:
            self._process_recognition(uploaded.getvalue())
        elif camera:
            self._process_recognition(camera.getvalue())

    def _process_recognition(self, image_bytes):
        matches = self.service.recognize_face(image_bytes)
        if matches:
            st.write(f"Найдено: {len(matches)}")
            for match in matches:
                st.image(match["image"], caption=f"Сходство: {match['similarity']:.2f}")
        else:
            st.error("Лицо не распознано")
