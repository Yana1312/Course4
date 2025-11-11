from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = FastAPI()


model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r", encoding="utf-8").readlines()


def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        contents = await image.read()

        image_data = Image.open(io.BytesIO(contents))
        if image_data.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image_data.size, (255, 255, 255))
            if image_data.mode == 'P':
                image_data = image_data.convert('RGBA')
            background.paste(image_data, mask=image_data.split()[-1] if image_data.mode == 'RGBA' else None)
            image_data = background
        else:
            image_data = image_data.convert('RGB')

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