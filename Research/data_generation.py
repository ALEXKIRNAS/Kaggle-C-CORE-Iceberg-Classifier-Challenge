from keras.preprocessing.image import ImageDataGenerator


def get_data_generator(X, y, batch_size=32):
    img_gen = ImageDataGenerator(
        rotation_range=0.,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.,
        zoom_range=0.,
        fill_mode='wrap',
        horizontal_flip=False,
        vertical_flip=True,
        data_format='channels_last')

    img_gen.fit(X)

    return img_gen.flow(X, y, batch_size=batch_size)
