# Face Recognition System

## Запуск проекта
1. Установите зависимости:  
   `pip install -r requirements.txt`

2. Запустите приложение:  
   `streamlit run app.py`

## Функционал
- Добавление лиц через файлы/архивы
- Распознавание через загрузку фото или камеру
- Хранение данных в SQLite

## Структура
- `app.py` - Точка входа
- `ui.py` - Интерфейс Streamlit
- `face_service.py` - Логика распознавания
- `database.py` - Работа с БД