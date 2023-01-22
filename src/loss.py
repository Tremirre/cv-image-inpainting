import tensorflow as tf
import tensorflow_addons as tfa


class MaskedMAE(tf.keras.losses.Loss):
    def __init__(self, name: str = "MaskAreaLoss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        rprod = tf.reduce_prod(y_true, axis=-1, keepdims=True)
        mask_present = tf.where(rprod == 0, 0.0, 1.0)
        return tf.reduce_mean(tf.abs(y_true - y_pred) * mask_present)


class MaskedGaussedSobelMAE(tf.keras.losses.Loss):
    def __init__(
        self, name: str = "MaskAreaLoss", kernel: tuple[int, int] = (3, 3), **kwargs
    ):
        self.kernel = kernel
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.image.sobel_edges(tfa.image.gaussian_filter2d(y_true, self.kernel))
        y_pred = tf.image.sobel_edges(tfa.image.gaussian_filter2d(y_pred, self.kernel))
        rprod = tf.reduce_prod(y_true, axis=-1, keepdims=True)
        mask_present = tf.where(rprod == 0, 0.0, 1.0)
        return tf.reduce_mean(tf.abs(y_true - y_pred) * mask_present)


class GaussedSobelMAE(tf.keras.losses.Loss):
    def __init__(
        self, name: str = "MaskAreaLoss", kernel: tuple[int, int] = (3, 3), **kwargs
    ):
        self.kernel = kernel
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.image.sobel_edges(tfa.image.gaussian_filter2d(y_true, self.kernel))
        y_pred = tf.image.sobel_edges(tfa.image.gaussian_filter2d(y_pred, self.kernel))
        return tf.reduce_mean(tf.abs(y_true - y_pred))


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(
        self, loss_dict: dict[str, tuple[tf.keras.losses.Loss, float]], **kwargs
    ):
        if not loss_dict:
            raise ValueError("loss_dict must not be empty")
        name = "+".join(
            f"{weight}*{loss_name}" for loss_name, (_, weight) in loss_dict.items()
        )
        self.loss_dict = loss_dict
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = tf.zeros(())
        for _, (loss_fn, weight) in self.loss_dict.items():
            loss += weight * loss_fn(y_true, y_pred)
        return loss
