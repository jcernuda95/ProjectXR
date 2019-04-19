import imageio
from keras import Model, Sequential
from keras.applications import DenseNet169
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import Sequence
from skimage import transform

from keras_applications.imagenet_utils import preprocess_input
from keras import optimizers
import sklearn
import random
import numpy as np
import pandas as pd


class MuraGenerator(Sequence):
    def __init__(self, path_to_file, batch_size, shuffle=True):
        self.bs = batch_size
        self.paths_images = pd.read_csv(path_to_file, names=["paths"])
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return self.paths_images.shape[0] // self.bs

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []

        while len(x_batch) < self.bs:
            path = str(self.paths_images['paths'].iloc[idx])
            image = imageio.imread(path)

            image = self.transform_image(image)

            x_batch.append(image)
            if "positive" in path:
                y_batch.append(1)
            else:
                y_batch.append(0)

        yield x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle is True:
            self.paths_images = sklearn.utils.shuffle(self.paths_images)

    def transform_image(self, image):
        image = transform.resize(image, (320, 320))

        image = preprocess_input(image)

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
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    Adam = optimizers.Adam(lr=0.0001)

    model.compile(optimizer=Adam,
                  metrics=['accuracy'],
                  loss='binary_crossentropy')

    return model


if __name__ == "__main__":
    train_generator = MuraGenerator(path_to_file='MURA-v1.1/train_image_paths.csv', batch_size=64)
    model = generate_model()

    csvlogger = CSVLogger('logs/training.log')
    checkpointer = ModelCheckpoint('models/model.hdf5', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                                  patience=2, min_lr=1e-6)

    model.fit_generator(train_generator,
                        callbacks=[csvlogger, checkpointer, reduce_lr],
                        class_weight={0: 0.3848, 1: 0.6152})
