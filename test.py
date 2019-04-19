import imageio
from keras import Model, Sequential
from keras.applications import DenseNet169
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import Sequence
from skimage import transform
from keras.preprocessing import image

from keras_applications.imagenet_utils import preprocess_input
from keras import optimizers
import sklearn
import random
import numpy as np
import os


class MuraGenerator(Sequence):
    def __init__(self, path_to_file, batch_size, shuffle=True):
        self.bs = batch_size
        self.paths_images = np.loadtxt(path_to_file, dtype='str')
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.paths_images) // self.bs

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []

        while len(x_batch) < self.bs:
            path = str(self.paths_images[idx])
            img = image.load_img(path, color_mode='rgb',
                                 target_size=(320, 320))

            img = image.img_to_array(img)

            img = self.transform_image(img)

            x_batch.append(img)
            y_batch.append(1 if "positive" in path else 0)

        return [np.asarray(x_batch), np.asarray(y_batch)]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.paths_images)

    def transform_image(self, image):

        image = preprocess_input(image, data_format='channels_last')

        if random.randint(0, 1):
            image = np.fliplr(image)

        angle = random.randint(-30, 30)
        image = transform.rotate(image, angle)

        return image


def generate_model():
    densenet = DenseNet169(include_top=False,
                           input_shape=(320, 320, 3),
                           weights='imagenet')

    for layer in densenet.layers:
        layer.trainable = False

    model = Sequential()

    model.add(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    adam = optimizers.Adam(lr=0.0001)

    model.compile(optimizer=adam,
                  metrics=['accuracy'],
                  loss='binary_crossentropy')

    return model


if __name__ == "__main__":
    model = generate_model()

    train_generator = MuraGenerator(
        path_to_file='MURA-v1.1/train_image_paths.csv', batch_size=8)

    if not os.path.isdir('logs'):
        os.mkdir('logs')

    csvlogger = CSVLogger('logs/training.log')
    checkpointer = ModelCheckpoint('models/model.hdf5', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                                  patience=2, min_lr=1e-6)

    model.fit_generator(train_generator,
                        verbose=1,
                        callbacks=[csvlogger, checkpointer, reduce_lr],
                        class_weight={0: 0.3848, 1: 0.6152})