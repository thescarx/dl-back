import tensorflow as tf
import base64
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
model = tf.keras.models.load_model("generator_2.h5")


@app.get("/")
async def index(request: Request):
    return 'hello world'


@app.get("/generate_images/{number}")
async def generate_images(number: int):

    noise = tf.random.normal([number, 100])
    images = model.predict(noise)

    images = np.uint8(np.clip((images + 1) / 2.0 * 255.0, 0, 255))
    images = [Image.fromarray(image) for image in images]

    # cas de telechargment ou trnasfert (conversion to text base64)
    image_b64s = []
    for image in images:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        image_b64s.append(base64.b64encode(image_bytes).decode("utf-8"))

    return {"images": image_b64s}
    # return {"images": "test api"}
