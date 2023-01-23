import tensorflow as tf


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection) / (tf.keras.backend.sum(y_true_f + y_pred_f))


def ssim_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.image.ssim(y_true, y_pred, max_val=1.0)
