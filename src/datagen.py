import tensorflow as tf
import numpy as np

from typing import Callable


class DatasetFillGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        image_size: tuple[int, int],
        channels: int,
        image_augmenter: Callable[[tf.Tensor], tf.Tensor],
        scale_max: float = 1.,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.image_size = image_size
        self.channels = channels
        self.image_augmenter = image_augmenter
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        for batch in self.dataset.skip(idx).take(1):
            source, target = self._process_batch(batch)
            break
        return source, target

    def _process_batch(self, batch):
        augmented = self.image_augmenter(batch)
        return augmented, batch

    def on_epoch_end(self):
        if self.shuffle:
            self.dataset = self.dataset.shuffle(len(self))