import os
import json
import tensorflow as tf
import logging
import base64

from flask import Flask, request, Response
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

ALL_MODELS = {}
for model in os.listdir("models"):
    ALL_MODELS[model] = tf.saved_model.load(os.path.join("models", model))


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    binary_jpg = data.get("image")
    mask_png = data.get("mask")
    model_name = data.get("model", "unet_working")
    used_model = ALL_MODELS.get(model_name)
    errors = {}
    if not used_model:
        errors["model"] = "Invalid model provided"
    if not binary_jpg:
        errors["image"] = "No image provided"
    if not mask_png:
        errors["mask"] = "No mask provided"
    if errors:
        return Response(
            json.dumps(errors),
            status=400,
            mimetype="application/json",
        )
    try:
        binary_jpg = base64.b64decode(binary_jpg)
        mask_png = base64.b64decode(mask_png)
        decoded_image = tf.io.decode_png(binary_jpg, channels=3)
        decoded_mask = tf.io.decode_png(mask_png, channels=3)
    except tf.errors.InvalidArgumentError:
        return Response(
            json.dumps({"error": "Invalid image provided - should be a png image"}),
            status=400,
            mimetype="application/json",
        )
    logging.info(f"Decoded image shape: {decoded_image.shape}")
    logging.info(f"Decoded mask shape: {decoded_mask.shape}")
    resized_mask = tf.image.resize(decoded_mask, (256, 256), antialias=False)
    resized_image = tf.image.resize(decoded_image, (256, 256)) / 255
    if "pconv" not in model_name:
        overlay_mask = tf.where(
            tf.reduce_sum(resized_mask, axis=-1, keepdims=True) > 0, 1.0, 0.0
        )
        resized_image = tf.where(overlay_mask == 1.0, 0, resized_image)
    else:
        overlay_mask = tf.where(
            tf.reduce_sum(resized_mask, axis=-1, keepdims=True) > 0, 0.0, 1.0
        )
        resized_image = tf.where(overlay_mask == 0.0, 0, resized_image)
    mask_size = tf.reduce_sum(overlay_mask)
    logging.info(f"Masked pixels: {mask_size} ({mask_size / 256 ** 2 * 100:.2f} %)")

    if "pconv" not in model_name:
        model_input = tf.concat([resized_image, overlay_mask], axis=-1)
        model_input = tf.reshape(model_input, (1, 256, 256, 4))
    else:
        mask_extended = tf.concat([overlay_mask] * 3, axis=-1)
        image = tf.reshape(resized_image, (1, 256, 256, 3))
        image = tf.cast(image, tf.float32)
        mask_extended = tf.reshape(mask_extended, (1, 256, 256, 3))
        mask_extended = tf.cast(mask_extended, tf.float32)
        model_input = [image, mask_extended]

    filled_image = used_model(model_input)
    filled_image = filled_image[0]
    filled_image = filled_image[:, :, :3] * 255
    filled_image = tf.cast(filled_image, tf.uint8)
    filled_image = tf.image.encode_jpeg(filled_image)
    return Response(filled_image.numpy(), mimetype="image/jpeg", status=200)


@app.route("/models")
def models():
    return Response(json.dumps(list(ALL_MODELS.keys())), mimetype="application/json")


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
