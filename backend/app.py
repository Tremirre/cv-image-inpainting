import json
import tensorflow as tf
import logging

from flask import Flask, request, Response

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

MODEL = tf.saved_model.load("models/unet_dummy")


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def predict():
    binary_jpeg = request.data
    if not binary_jpeg:
        return Response(
            json.dumps({"error": "No image provided"}),
            status=400,
            mimetype="application/json",
        )
    try:
        decoded_image = tf.io.decode_jpeg(binary_jpeg)
    except tf.errors.InvalidArgumentError:
        return Response(
            json.dumps({"error": "Invalid image provided - should be a jpg image"}),
            status=400,
            mimetype="application/json",
        )
    logging.info(f"Decoded image shape: {decoded_image.shape}")
    resized_image = tf.image.resize(decoded_image, (256, 256)) / 255
    overlay_mask = tf.where(
        tf.reduce_sum(resized_image, axis=-1, keepdims=True) == 3, 1.0, 0.0
    )
    mask_size = tf.reduce_sum(overlay_mask)
    logging.info(f"Masked pixels: {mask_size} ({mask_size / 256 ** 2 * 100:.2f} %)")

    resized_image = tf.concat([resized_image, overlay_mask], axis=-1)
    reshaped_image = tf.reshape(resized_image, (1, 256, 256, 4))
    logging.info(f"Reshaped image shape: {reshaped_image.shape}")
    filled_image = MODEL(reshaped_image)
    filled_image = tf.reshape(filled_image, (256, 256, 4))
    filled_image = filled_image[:, :, :3] * 255
    filled_image = tf.cast(filled_image, tf.uint8)
    filled_image = tf.image.encode_jpeg(filled_image)
    return Response(filled_image.numpy(), mimetype="image/jpeg", status=200)


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
