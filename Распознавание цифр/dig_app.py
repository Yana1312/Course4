import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Распознавание цифр",
    layout="centered"
)

st.title("Распознавание рукописных цифр")

API_ENDPOINT = "http://localhost:8000/predict"

if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

try:
    from streamlit_drawable_canvas import st_canvas

    col1, col2 = st.columns([2, 1])

    with col1:
        brush_size = st.slider("Размер кисти", 10, 30, 15)

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=brush_size,
            stroke_color="rgba(0, 0, 0, 1)",
            background_color="rgba(255, 255, 255, 1)",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True,
        )

        if canvas_result.image_data is not None:
            st.session_state.canvas_data = canvas_result.image_data

        if st.button("Распознать цифру"):
            if st.session_state.canvas_data is not None:
                with st.spinner("Анализ..."):
                    try:
                        img_array = st.session_state.canvas_data
                        if img_array.shape[-1] == 4:
                            img_array = img_array[:, :, :3]
                        image = Image.fromarray(img_array.astype('uint8'))

                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='PNG')
                        img_bytes.seek(0)

                        files = {"image": ("digit.png", img_bytes, "image/png")}
                        response = requests.post(API_ENDPOINT, files=files)

                        if response.status_code == 200:
                            st.session_state.prediction_result = response.json()
                        else:
                            st.error("Ошибка сервера")

                    except requests.exceptions.ConnectionError:
                        st.error("Сервер не доступен")
                    except Exception as e:
                        st.error("Ошибка обработки")

    with col2:
        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result

            st.metric(
                label="Цифра",
                value=result['class'],
            )

            st.metric(
                label="Уверенность",
                value=f"{result['confidence']:.1%}",
            )

        else:
            st.write("Результат")

except ImportError:
    st.error("Установите streamlit-drawable-canvas")