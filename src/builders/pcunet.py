import tensorflow as tf

from typing import Type


class PCUNETBuilder:
    def __init__(
        self,
        input_shape,
        output_shape,
        pconv_class: Type[tf.keras.layers.Layer],
        n_filters=64,
        n_blocks=4,
        activation="relu",
        last_activation="sigmoid",
        dropout_rate=0,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pconv_class = pconv_class
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.activation = activation
        self.last_activation = last_activation
        self.dropout_rate = dropout_rate

    def build(self):
        inputs = [
            tf.keras.Input(shape=self.input_shape),
            tf.keras.Input(shape=self.input_shape),
        ]
        convs = []
        masks = []
        for i in range(self.n_blocks):
            last_layer, last_mask = inputs if i == 0 else (convs[-1], masks[-1])
            c1, m1, c2, m2 = self._add_encode_block(
                last_layer, last_mask, self.n_filters * (2**i)
            )
            convs.extend([c1, c2])
            masks.extend([m1, m2])

        for i in range(self.n_blocks):
            conv_prev = convs[2 * self.n_blocks - 2 * (i + 1)]
            mask_prev = masks[2 * self.n_blocks - 2 * (i + 1)]
            filters_l = self.n_filters * (2 ** (self.n_blocks - i - 1))
            filters_r = self.n_filters * (2 ** (self.n_blocks - i - 2))
            c1, m1, c2, m2 = self._add_decode_block(
                convs[-1], masks[-1], conv_prev, mask_prev, filters_l, filters_r
            )
            convs.extend([c1, c2])
            masks.extend([m1, m2])

        outputs = tf.keras.layers.Conv2D(
            self.output_shape[-1],
            (3, 3),
            activation=self.last_activation,
            padding="same",
        )(convs[-1])
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _add_encode_block(self, last_layer, last_mask, num_filters):
        conv1, mask1 = self.pconv_class(
            num_filters, (3, 3), strides=1, padding="same", activation=self.activation
        )([last_layer, last_mask])
        if self.dropout_rate > 0:
            conv1 = tf.keras.layers.Dropout(self.dropout_rate)(conv1)
        conv2, mask2 = self.pconv_class(
            num_filters, (3, 3), strides=2, padding="same", activation=self.activation
        )([conv1, mask1])

        return conv1, mask1, conv2, mask2

    def _add_decode_block(
        self,
        last_layer,
        last_mask,
        share_layer,
        share_mask,
        num_filters_1,
        num_filters_2,
    ):
        up_layer = tf.keras.layers.UpSampling2D(size=(2, 2))(last_layer)
        up_mask = tf.keras.layers.UpSampling2D(size=(2, 2))(last_mask)

        concat_layer = tf.keras.layers.Concatenate(axis=3)([share_layer, up_layer])
        concat_mask = tf.keras.layers.Concatenate(axis=3)([share_mask, up_mask])

        conv1, mask1 = self.pconv_class(
            num_filters_1, (3, 3), padding="same", activation=self.activation
        )([concat_layer, concat_mask])
        conv2, mask2 = self.pconv_class(
            num_filters_2, (3, 3), padding="same", activation=self.activation
        )([conv1, mask1])

        return conv1, mask1, conv2, mask2
