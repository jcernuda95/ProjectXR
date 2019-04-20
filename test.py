import argparse
import os
import random
from glob import glob

import keras as keras
import numpy as np
from keras import backend as K
from keras import Sequential
from keras import optimizers
from keras.applications import DenseNet169
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import Sequence
from keras_applications.imagenet_utils import preprocess_input
from skimage import transform
from sklearn.metrics import precision_score, recall_score

weights = []


def recall(y_true, y_pred):
    global weights
    return K.tf.metrics.recall(y_true, y_pred, weights)


def precision(y_true, y_pred):
    global weights
    return K.tf.metrics.precision(y_true, y_pred, weights)


def auc(y_true, y_pred):
    global weights
    return K.tf.metrics.auc(y_true, y_pred, weights)


class MuraGenerator(Sequence):
    def __init__(self, paths_images, batch_size, weights, augment=False):
        self.bs = batch_size
        self.paths_images = paths_images
        self.augment = augment
        self.weights = weights

    def __len__(self):
        return len(self.paths_images) // self.bs

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        w_batch = []

        while len(x_batch) < self.bs:
            path = str(self.paths_images[idx])
            img = image.load_img(path, color_mode='rgb',
                                 target_size=(320, 320))

            img = image.img_to_array(img)

            img = self.transform_image(img)

            x_batch.append(img)
            y_batch.append(1 if "positive" in path else 0)
            w_batch.append(self.weights[1] if "positive" in path else self.weights[0])

        return [np.asarray(x_batch), np.asarray(y_batch), np.asarray(w_batch)]

    def transform_image(self, image):

        image = preprocess_input(image, data_format='channels_last')

        if self.augment:
            if random.randint(0, 1):
                image = np.fliplr(image)

            angle = random.randint(-30, 30)
            image = transform.rotate(image, angle)

        return image


def generate_model(stage):
    densenet = DenseNet169(include_top=False,
                           input_shape=(320, 320, 3),
                           weights='imagenet')

    for layer in densenet.layers:
        layer.trainable = False
    if stage >= 1:
        for layer in densenet.layers:
            if 'conv5' in layer.name:
                layer.trainable = True

    model = Sequential()

    model.add(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    adam = optimizers.Adam(lr=1e-3)

    model.compile(optimizer=adam,
                  metrics=['accuracy', recall, precision, auc],
                  loss='binary_crossentropy')

    return model


if __name__ == "__main__":
    global weights

    parser = argparse.ArgumentParser(description="MURA image classification")
    parser.add_argument('-r', '--resume', action='store_true', default='False',
                        help='Resume training from last saved model')
    parser.add_argument('-s', '--stage', default=0,
                        help='Set stage of training: '
                             '0-train only dense layer.'
                             '1-train only dense layer with image augmentation.'
                             '2-train dense with augmentation and last conv block.')
    parser.add_argument('--train_path', default='./MURA-v1.1/train_image_paths.csv',
                        help='Path to train_image_paths.csv')
    parser.add_argument('--test_path', default='./MURA-v1.1/valid_image_paths.csv',
                        help='Path to test_image_paths.csv')
    args = parser.parse_args()

    starting_epoch = 0

    img_paths = np.loadtxt(args.train_path, dtype='str')
    img_paths = [str(i) for i in img_paths]
    positives = 0
    for path in img_paths:
        positives += 1 if "positive" in path else 0
    negatives = len(img_paths) - positives

    total = float(len(img_paths))
    weights = [negatives/total, positives/total]

    print("Weights: ", weights)

    if args.resume is True:
        paths_models = sorted(glob('models/*'))
        model = load_model(paths_models[-1])
        starting_epoch = int(paths_models[-1][7:9])
        print("Path: ", paths_models[-1])
        print("starting epoch: ", int(paths_models[-1][7:9]))
    else:
        model = generate_model(int(args.stage))

    val_split = int(0.75 * len(img_paths))
    train_paths = img_paths[:val_split]
    val_paths = img_paths[val_split:]

    train_generator = MuraGenerator(train_paths, batch_size=16, weights=weights, augment=True if int(args.stage) > 0 else False)
    val_generator = MuraGenerator(val_paths, batch_size=16, weights=weights)

    if not os.path.isdir('logs'):
        os.mkdir('logs')

    csvlogger = CSVLogger('logs/training.log', append=args.resume)
    checkpointer = ModelCheckpoint(
        'models/model-e{epoch:2d}.hdf5', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                                  patience=1, min_lr=1e-6)

    model.fit_generator(train_generator,
                        callbacks=[csvlogger, checkpointer, reduce_lr],
                        epochs=10,
                        initial_epoch=starting_epoch,
                        validation_data=val_generator)