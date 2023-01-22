import tensorflow as tf
import numpy as np

from typing import Callable


def identical_augmenter(images: tf.Tensor) -> tf.Tensor:
    new = tf.identity(images)
    return new, images


def mask_image_augmenter(
    images: tf.Tensor, mask_generator: Callable[[tf.Tensor], np.ndarray], max_val: float
) -> tf.Tensor:
    masks = mask_generator(images.shape[0])
    augmented = tf.where(masks == 0, images, max_val)
    return augmented, images


def masked_channel_augmenter(
    images: tf.Tensor, mask_generator: Callable[[tf.Tensor], np.ndarray]
) -> tf.Tensor:
    masks = mask_generator(images.shape[0])
    augmented = tf.where(masks == 0, images, 0.0)
    masks = tf.where(masks == 0, 0.0, 1.0)
    augmented = tf.concat([augmented, masks[:, :, :, :1]], axis=-1)
    images = tf.concat([images, masks[:, :, :, :1]], axis=-1)
    return augmented, images
