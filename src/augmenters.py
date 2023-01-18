import tensorflow as tf
import numpy as np

from typing import Callable


def identical_augmenter(x: tf.Tensor) -> tf.Tensor:
    new = tf.identity(x)
    return new

def mask_image_augment(images: tf.Tensor, mask_generator: Callable[[tf.Tensor], np.ndarray], max_val: float) -> tf.Tensor:
    masks = mask_generator(images.shape[0])
    augmented = tf.where(masks == 0, images, max_val) 
    return augmented