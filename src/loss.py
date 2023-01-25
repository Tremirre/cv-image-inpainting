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


class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self, name: str = "SSIMLoss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


class ExcessivePixelDiffLoss(tf.keras.losses.Loss):
    def __init__(self, name: str = "ExcessiveVariationLoss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_var_x = tf.reduce_mean(tf.abs(y_true[:, 1:, :, :] - y_true[:, :-1, :, :]))
        true_var_y = tf.reduce_mean(tf.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :]))
        pred_var_x = tf.reduce_mean(tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]))
        pred_var_y = tf.reduce_mean(tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
        dvar_X = pred_var_x - true_var_x
        dvar_Y = pred_var_y - true_var_y
        return tf.maximum(dvar_X, 0) + tf.maximum(dvar_Y, 0)


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(
        self, loss_dict: dict[str, tuple[tf.keras.losses.Loss, float]], **kwargs
    ):
        if not loss_dict:
            raise ValueError("loss_dict must not be empty")
        name = "_".join(
            f"{weight}{loss_name}" for loss_name, (_, weight) in loss_dict.items()
        )
        self.loss_dict = loss_dict
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = tf.zeros(())
        for _, (loss_fn, weight) in self.loss_dict.items():
            loss += weight * loss_fn(y_true, y_pred)
        return loss
