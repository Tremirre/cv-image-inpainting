import tensorflow as tf
import numpy as np

from typing import Callable


class DatasetFillGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        image_size: tuple[int, int],
        channels: int,
        mask_generator: Callable[[int], np.ndarray],
        scale_max: float = 1.
    ) -> None:
        self.dataset = dataset
        self.image_size = image_size
        self.channels = channels
        self.mask_generator = mask_generator
        self.scale_max = scale_max

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.dataset.skip(idx)
        batch = next(iter(self.dataset))
        source, target = self._process_batch(batch)
        return source, target

    def _process_batch(self, batch):
        effective_batch_size = batch.shape[0]
        masks = self.mask_generator(effective_batch_size)
        target = tf.identity(batch)
        batch = tf.where(masks == 0, batch, self.scale_max)
        return batch, target
