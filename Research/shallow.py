from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalAveragePooling2D, PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.regularizers import l2


def conv_bn_activation(input_value,
                       filters, 
                       kernel_size=(3, 3), 
                       strides=(1, 1), 
                       weight_decay=1e-4):
    
    x = BatchNormalization()(input_value)
    x = Conv2D(filters, 
               kernel_size,
               strides=strides,
               padding='VALID',
               use_bias=True,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x


def get_shallow_model(input_shape=(75, 75, 3), weight_decay=1e-4):
    #Building the model
    input = Input(shape=input_shape)
    
    #Conv Layer 1
    x = conv_bn_activation(input, filters=16, weight_decay=weight_decay)
    x = conv_bn_activation(x, filters=16, kernel_size=(1, 1), strides=(1, 1), weight_decay=weight_decay)

    #Conv Layer 2
    x = conv_bn_activation(x, filters=32, weight_decay=weight_decay)
    x = conv_bn_activation(x, filters=32, kernel_size=(1, 1), strides=(1, 1), weight_decay=weight_decay)

    #Conv Layer 3
    x = conv_bn_activation(x, filters=32, weight_decay=weight_decay)
    x = conv_bn_activation(x, filters=32, kernel_size=(3, 3), strides=(2, 2), weight_decay=weight_decay)

    #Conv Layer 4
    x = conv_bn_activation(x, filters=64, weight_decay=weight_decay)
    x = conv_bn_activation(x, filters=64, kernel_size=(1, 1), strides=(1, 1), weight_decay=weight_decay)
    
    #Conv Layer 5
    x = conv_bn_activation(x, filters=64, weight_decay=weight_decay)
    x = conv_bn_activation(x, filters=64, kernel_size=(3, 3), strides=(2, 2), weight_decay=weight_decay)

    #Conv Layer 6
    x = conv_bn_activation(x, filters=128, weight_decay=weight_decay)
    x = conv_bn_activation(x, filters=128, kernel_size=(1, 1), strides=(1, 1), weight_decay=weight_decay)
    
    #Conv Layer 7
    x = conv_bn_activation(x, filters=128, weight_decay=weight_decay)
    x = conv_bn_activation(x, filters=128, kernel_size=(3, 3), strides=(2, 2), weight_decay=weight_decay)

    #Global pooling
    x = conv_bn_activation(x, filters=256, kernel_size=(1, 1), weight_decay=weight_decay)
    x = GlobalAveragePooling2D()(x)

    #Sigmoid Layer
    outputs = Dense(1, 
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(weight_decay), 
                    activation='sigmoid')(x)
    model = Model(inputs=input, outputs=outputs)
    
    return model
