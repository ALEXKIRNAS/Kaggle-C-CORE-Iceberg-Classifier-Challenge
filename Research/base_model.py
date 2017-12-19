from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D, ActivityRegularization
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop


def get_base_model():
    WEIGHT_DECAY = 0
    
    #Building the model
    model=Sequential()
    
    #Conv Layer 1
    model.add(Conv2D(64, 
                     kernel_size=(3, 3), 
                     activation='relu',
                     kernel_regularizer=l2(WEIGHT_DECAY),
                     input_shape=(75, 75, 2)))
    
    model.add(MaxPooling2D(pool_size=(3, 3), 
                           strides=(2, 2)))

    #Conv Layer 2
    model.add(Conv2D(128, kernel_size=(3, 3), 
                     activation='relu',
                     kernel_regularizer=l2(WEIGHT_DECAY)))
    
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    #Conv Layer 3
    model.add(Conv2D(128, 
                     kernel_size=(3, 3), 
                     activation='relu',
                     kernel_regularizer=l2(WEIGHT_DECAY)))
    
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    #Conv Layer 4
    model.add(Conv2D(256, 
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    #Flatten the data for upcoming dense layers
    model.add(Flatten())

    #Dense Layers
    model.add(Dense(256, kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(Activation('relu'))

    #Dense Layer 2
    model.add(Dense(128, kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(Activation('relu'))

    #Sigmoid Layer
    model.add(Dense(2, kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(Activation('softmax'))

    opt=SGD(lr=0.002, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model
