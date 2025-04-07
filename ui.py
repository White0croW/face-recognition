import zipfile
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
            type=["jpg", "png", "zip"],  # Streamlit автоматически распознаёт ZIP [[2]]
            accept_multiple_files=True,
        )

        if files:
            total_images = 0
            valid_images = 0
            errors = []

            # Подсчёт валидных изображений с учётом MIME-типов ZIP
            for file in files:
                if file.type in ["application/zip", "application/x-zip-compressed"]:
                    try:
                        with zipfile.ZipFile(file) as z:
                            # Игнорируем служебные файлы macOS и папки [[7]]
                            filelist = [
                                f
                                for f in z.namelist()
                                if not f.startswith("__MACOSX/")
                                and not f.endswith("/")
                                and f.lower().endswith((".jpg", ".png"))
                            ]
                            total_images += len(filelist)
                    except zipfile.BadZipFile:
                        errors.append(
                            f"Файл {file.name} не является валидным ZIP-архивом"
                        )
                else:
                    total_images += 1

            progress_bar = st.progress(0)
            processed = 0

            for file in files:
                try:
                    if file.type in ["application/zip", "application/x-zip-compressed"]:
                        with zipfile.ZipFile(file) as z:
                            for name in z.namelist():
                                if (
                                    not name.startswith("__MACOSX/")
                                    and not name.endswith("/")
                                    and name.lower().endswith((".jpg", ".png"))
                                ):
                                    img_bytes = z.read(name)
                                    self.service.add_image_to_db(img_bytes)
                                    valid_images += 1
                    else:
                        self.service.add_image_to_db(file.getvalue())
                        valid_images += 1
                except Exception as e:
                    errors.append(f"Ошибка в файле {file.name}: {str(e)}")
                finally:
                    processed += 1
                    progress_bar.progress(processed / total_images)

            progress_bar.empty()
            st.success(f"✅ Загружено изображений: {valid_images}")
            if errors:
                st.error(f"❌ Ошибок: {len(errors)}")
                with st.expander("Подробнее"):
                    st.write("\n".join(errors))

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
