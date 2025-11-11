from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = FastAPI()


model = load_model("model12.h5", compile=False)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'L':
        image = image.convert('L')

    size = (28, 28)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = image_array.astype(np.float32) / 255.0

    processed_image = normalized_image_array.reshape(1, 28, 28)

    return processed_image


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        contents = await image.read()

        image_data = Image.open(io.BytesIO(contents))

        processed_image = preprocess_image(image_data)
        prediction = model.predict(processed_image)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])

        return {
            "class": class_name,
            "confidence": confidence_score,
            "class_index": int(index)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")