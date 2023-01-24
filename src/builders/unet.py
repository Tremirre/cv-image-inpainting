import tensorflow as tf


class UNETBuilder:
    def __init__(
        self,
        input_shape,
        output_shape,
        n_filters=64,
        n_blocks=4,
        n_convs=2,
        activation="relu",
        last_activation="sigmoid",
        dropout_rate=0,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.n_convs = n_convs
        self.activation = activation
        self.last_activation = last_activation
        self.dropout_rate = dropout_rate

    def build(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        skips = []
        for i in range(self.n_blocks):
            for _ in range(self.n_convs):
                x = tf.keras.layers.Conv2D(
                    self.n_filters * 2**i,
                    3,
                    padding="same",
                    activation=self.activation,
                )(x)
            skips.append(x)
            x = tf.keras.layers.MaxPool2D(2)(x)
            if i > 0 and self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        for i in range(self.n_blocks):
            x = tf.keras.layers.Conv2D(
                self.n_filters * 2 ** (self.n_blocks - i - 1),
                3,
                padding="same",
                activation=self.activation,
            )(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Concatenate()([x, skips[self.n_blocks - i - 1]])
            for _ in range(self.n_convs):
                x = tf.keras.layers.Conv2D(
                    self.n_filters * 2 ** (self.n_blocks - i - 1),
                    3,
                    padding="same",
                    activation=self.activation,
                )(x)
        outputs = tf.keras.layers.Conv2D(
            self.output_shape[-1], 1, padding="same", activation=self.last_activation
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
