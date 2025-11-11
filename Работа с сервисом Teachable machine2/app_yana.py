import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Классификатор обуви",
    page_icon="👟",
    layout="centered"
)

st.title("Классификатор обуви")
st.markdown("---")

API_ENDPOINT = "http://localhost:8000/predict"

with st.sidebar:
    st.subheader("О приложении")
    st.info("""
    Классификатор обуви - это инструмент 
    для автоматического определения типа обуви
    на изображениях с использованием 
    искусственного интеллекта.

    Как использовать:
    1. Загрузите изображение обуви 
    2. Нажмите кнопку анализа
    3. Получите результат классификации

    Технические детали:
    - Количество классов: 3 типа обуви
    """)

uploaded_image = st.file_uploader(
    "Загрузите изображение обуви",
    type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
    help="Выберите изображение в одном из поддерживаемых форматов"
)

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if uploaded_image is not None:
    image_data = Image.open(uploaded_image)
    st.session_state.current_image = image_data

    st.image(image_data, caption="Загруженное изображение", width=400)

    if st.button("Какая же эта туфелька?", type="primary", use_container_width=True):
        with st.spinner("Бежим на бал...Подождите пожалуйста"):
            image_buffer = io.BytesIO()

            if image_data.mode in ('RGBA', 'LA', 'P'):
                converted_image = image_data.convert('RGB')
                converted_image.save(image_buffer, format='JPEG', quality=95)
            else:
                image_data.save(image_buffer, format='JPEG', quality=95)

            image_buffer.seek(0)

            files = {"image": ("image.jpg", image_buffer, "image/jpeg")}
            api_response = requests.post(API_ENDPOINT, files=files)

            if api_response.status_code == 200:
                st.session_state.analysis_result = api_response.json()
                st.success("Прибежали!")
            else:
                st.error(f"Ошибка API: {api_response.text}")

else:
    st.info("Загрузите изображение для анализа")

st.markdown("---")

if st.session_state.analysis_result is not None:
    result_data = st.session_state.analysis_result

    st.subheader("Результаты классификации")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Класс обуви")
        st.info(result_data['class'])


    with col2:
        st.write("Идентификатор класса")
        st.info(result_data['class_index'])

else:
    st.info("Результаты анализа появятся здесь после обработки изображения")