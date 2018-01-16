import six
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    GaussianNoise,
    Lambda,
    concatenate
)
from keras.layers.convolutional import Conv2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import GlobalAveragePooling2D

WEIGHT_DECAY = None


def _bn_elu(input):
    """Helper to build a BN -> elu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return ELU()(norm)


def _conv_bn_elu(**conv_params):
    """Helper to build a conv -> BN -> elu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(WEIGHT_DECAY))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      use_bias=True)(input)
        return _bn_elu(conv)

    return f


def _bn_elu_conv(**conv_params):
    """Helper to build a BN -> elu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(WEIGHT_DECAY))

    def f(input):
        activation = _bn_elu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      use_bias=True)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(WEIGHT_DECAY),
                          use_bias=True)(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->elu since we just did bn->elu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(WEIGHT_DECAY),
                           use_bias=True)(input)
        else:
            conv1 = _bn_elu_conv(filters=filters, kernel_size=(3, 3),
                                 strides=init_strides)(input)

        residual = _bn_elu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(WEIGHT_DECAY))(input)
        else:
            conv_1_1 = _bn_elu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_elu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_elu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, filters, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)

        x = Lambda(lambda x: x[:, :, :, 0:2]
                             if K.image_data_format() == 'channels_last'
                             else x[:, 0:2, :, :])(input)

        angle = Lambda(lambda x: x[:, :, :, 2:]
                              if K.image_data_format() == 'channels_last'
                              else x[:, 2:, :, :])(input)

        x_noise = GaussianNoise(3e-1)(x)

        noise_input = concatenate([x_noise, angle], axis=-1)

        conv1 = _conv_bn_elu(filters=filters, kernel_size=(7, 7), strides=(2, 2))(noise_input)

        block = conv1
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_elu(block)

        # Classifier block
        pool2 = GlobalAveragePooling2D(data_format='channels_last')(block)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal", use_bias=True,
                      kernel_regularizer=l2(WEIGHT_DECAY), activation="softmax")(pool2)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, filters, weight_decay=0., weights=None):
        global WEIGHT_DECAY
        WEIGHT_DECAY = weight_decay 

        model = ResnetBuilder.build(input_shape, num_outputs, basic_block, filters, [2, 2, 2, 2])

        if weights:
            model.load_weights(weights)

        return model

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, filters, weight_decay=0., weights=None):
        global WEIGHT_DECAY
        WEIGHT_DECAY = weight_decay

        return ResnetBuilder.build(input_shape, num_outputs, basic_block, filters, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, filters, weight_decay=0., weights=None):
        global WEIGHT_DECAY
        WEIGHT_DECAY = weight_decay

        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, filters, [3, 4, 6, 3])