from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dense, add, GaussianNoise
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
selu = "selu_"


# Modular function for Fire Node
def fire_module(x, fire_id, squeeze=16, expand=64, weight_decay=0.):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=True,
                      name=s_id + sq1x1)(x)
    x = Activation('selu', name=s_id + selu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid',
                         kernel_regularizer=l2(weight_decay),
                         use_bias=True,
                         name=s_id + exp1x1)(x)
    left = Activation('selu', name=s_id + selu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same',
                          kernel_regularizer=l2(weight_decay),
                          use_bias=True,
                          name=s_id + exp3x3)(x)
    right = Activation('selu', name=s_id + selu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

def SqueezeNet(
        input_shape=None,
        filters=8,
        weight_decay=0.,
        classes=2):
    """Instantiates the SqueezeNet architecture.
    """

    assert filters % 2 == 0, 'Number of filters must be 2*n, n > 1'

    img_input = Input(shape=input_shape)

    x = Convolution2D(filters, (3, 3), padding='valid',
                      use_bias=True,
                      kernel_regularizer=l2(weight_decay),
                      name='conv1')(img_input)
    x = Activation('selu', name='selu_conv1')(x)

    x = fire_module(x, fire_id=2, squeeze=filters // 2, expand=filters, weight_decay=weight_decay)
    residual = x
    x = fire_module(x, fire_id=3, squeeze=filters // 2, expand=filters, weight_decay=weight_decay)
    x = add([x, residual])
    filters *= 2
    x = fire_module(x, fire_id=4, squeeze=filters // 2, expand=filters, weight_decay=weight_decay)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    residual = x
    x = fire_module(x, fire_id=5, squeeze=filters // 2, expand=filters, weight_decay=weight_decay)
    x = add([x, residual])
    filters *=2
    x = fire_module(x, fire_id=6, squeeze=filters // 2, expand=filters, weight_decay=weight_decay)
    residual = x
    x = fire_module(x, fire_id=7, squeeze=filters // 2, expand=filters, weight_decay=weight_decay)
    x = add([x, residual])
    filters *= 2
    x = fire_module(x, fire_id=8, squeeze=filters // 2, expand=filters, weight_decay=weight_decay)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(x)

    residual = x
    x = fire_module(x, fire_id=9, squeeze=filters // 2, expand=filters, weight_decay=weight_decay)
    x = add([x, residual])
    x = Convolution2D(filters, (1, 1), padding='valid',
                      use_bias=True,
                      kernel_regularizer=l2(weight_decay),
                      name='conv10')(x)
    x = Activation('selu', name='selu_conv10')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input
    model = Model(inputs, x, name='squeezenet')

    return model
