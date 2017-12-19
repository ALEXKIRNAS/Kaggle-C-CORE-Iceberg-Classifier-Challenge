from keras.preprocessing.image import ImageDataGenerator

def get_data_generator(X, y, batch_size=32):
    img_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0.,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.,
        zoom_range=0.,
        fill_mode='wrap',
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None,
        data_format='channels_last')
    
    img_gen.fit(X)
    
    return img_gen.flow(X, y, batch_size=batch_size)
