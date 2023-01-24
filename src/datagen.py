import tensorflow as tf

from typing import Callable, TypeVar

T = TypeVar("T")


class DatasetFillGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        image_augmenter: Callable[[tf.Tensor], tuple[T, tf.Tensor]],
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.image_augmenter = image_augmenter
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[T, tf.Tensor]:
        for batch in self.dataset.skip(idx).take(1):
            return self._process_batch(batch)

    def _process_batch(self, batch: tf.Tensor) -> tuple[T, tf.Tensor]:
        augmented, batch = self.image_augmenter(batch)
        return augmented, batch

    def on_epoch_end(self):
        if self.shuffle:
            self.dataset = self.dataset.shuffle(len(self))
