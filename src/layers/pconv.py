# Original Author: Ayush Thakur - https://github.com/ayulockin/deepimageinpainting
import tensorflow as tf


class PConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [
            tf.keras.layers.InputSpec(ndim=4),
            tf.keras.layers.InputSpec(ndim=4),
        ]

    def build(self, input_shape):
        """Adapted from original _Conv() layer of Keras
        param input_shape: list of dimensions for [img, mask]
        """
        channel_axis = -1
        if self.data_format == "channels_first":
            channel_axis = 1

        if input_shape[0][channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. Found `None`."
            )

        self.input_dim = input_shape[0][channel_axis]

        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="img_kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.kernel_mask = tf.keras.backend.ones(
            shape=self.kernel_size + (self.input_dim, self.filters)
        )

        # Calculate padding size to achieve zero-padding
        padding_dim = int((self.kernel_size[0] - 1) / 2)
        self.pconv_padding = (
            (padding_dim, padding_dim),
            (padding_dim, padding_dim),
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs):
        """
        We will be using the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we clip mask values to between 0 and 1
        """
        if len(inputs) != 2:
            raise ValueError(
                "PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: "
                + str(inputs)
            )

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = tf.keras.backend.spatial_2d_padding(
            inputs[0], self.pconv_padding, self.data_format
        )
        masks = tf.keras.backend.spatial_2d_padding(
            inputs[1], self.pconv_padding, self.data_format
        )

        # Apply convolutions to mask
        mask_output = tf.keras.backend.conv2d(
            masks,
            self.kernel_mask,
            strides=self.strides,
            padding="valid",
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Apply convolutions to image
        img_output = tf.keras.backend.conv2d(
            (images * masks),
            self.kernel,
            strides=self.strides,
            padding="valid",
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)
        mask_output = tf.keras.backend.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        if self.use_bias:
            img_output = tf.keras.backend.bias_add(
                img_output, self.bias, data_format=self.data_format
            )
        if self.activation is not None:
            img_output = self.activation(img_output)

        return img_output, mask_output

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            space = input_shape[0][1:-1]
        elif self.data_format == "channels_first":
            space = input_shape[0][2:]
        else:
            raise ValueError("Invalid data_format: " + self.data_format)

        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size[i],
                padding="same",
                stride=self.strides[i],
                dilation=self.dilation_rate[i],
            )
            new_space.append(new_dim)

        new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
        if self.data_format == "channels_first":
            new_shape = (input_shape[0][0], self.filters) + tuple(new_space)
        return new_shape, new_shape


## Reference: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/conv_utils.py#L85
def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.
    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.
    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {"same", "valid", "full", "causal"}
    dilated_filter_size = (filter_size - 1) * dilation + 1
    if padding == "same":
        output_length = input_length
    elif padding == "valid":
        output_length = input_length - dilated_filter_size + 1
    elif padding == "causal":
        output_length = input_length
    elif padding == "full":
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride
