import argparse
import os
import random
from glob import glob

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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, log_loss, accuracy_score

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    print("No MatplotLib")
    pass

def recall(y_true, y_pred, weights):
    return recall_score(y_true, y_pred, sample_weight=weights)


def precision(y_true, y_pred, weights):
    return precision_score(y_true, y_pred, sample_weight=weights)


def auc(y_true, y_pred, weights):
    return roc_auc_score(y_true, y_pred, sample_weight=weights)


def transform_image(image, augment):
    image = preprocess_input(image, data_format='channels_last')

    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)

        angle = random.randint(-30, 30)
        image = transform.rotate(image, angle)

    return image


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def getCAM(model, img):
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "conv5_3")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])

    #TODO : Understand this
    target_class = 1
    for i, w in enumerate(class_weights[:, target_class]):
        cam += w * conv_outputs[i, :, :]

    return cam


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

            img = transform_image(img, self.augment)

            x_batch.append(img)
            y_batch.append(1 if "positive" in path else 0)
            w_batch.append(self.weights[1] if "positive" in path else self.weights[0])

        return [np.asarray(x_batch), np.asarray(y_batch), np.asarray(w_batch)]


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
                  metrics=['accuracy'],
                  loss='binary_crossentropy')

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MURA image classification")
    parser.add_argument('-r', '--resume', action='store_true', default='False',
                        help='Resume training from last saved model')
    parser.add_argument('-s', '--stage', default=0,
                        help='Set stage of training: '
                             '0-train only dense layer.'
                             '1-train only dense layer with image augmentation.'
                             '2-train dense with augmentation and last conv block.'
                             '3-testing.')
    parser.add_argument('--train_path', default='./MURA-v1.1/train_image_paths.csv',
                        help='Path to train_image_paths.csv')
    parser.add_argument('--test_path', default='./MURA-v1.1/valid_image_paths.csv',
                        help='Path to test_image_paths.csv')
    parser.add_argument('--model_path', help='Path to a model to resume or proceed with transfer learning')
    parser.add_argument('-c', '--client', default=0,
                        help='Client to evaluate')
    args = parser.parse_args()

    starting_epoch = 0

    model = generate_model(int(args.stage))

    if args.resume is True or int(args.stage) == 2:
        model.load_weights(args.model_path)
        print("Path: ", args.model_path)
        if args.resume is True:
            starting_epoch = int(args.model_path[31:33])
            print("starting epoch: ", starting_epoch)

    img_paths = np.loadtxt(args.train_path, dtype='str')
    img_paths = [str(i) for i in img_paths]
    positives = 0
    for path in img_paths:
        positives += 1 if "positive" in path else 0
    negatives = len(img_paths) - positives

    total = float(len(img_paths))
    weights = [negatives/total, positives/total]

    # weights = [1,1]
    print("Weights: ", weights)

    if int(args.stage) < 3:
        val_split = int(0.75 * len(img_paths))
        train_paths = img_paths[:val_split]
        val_paths = img_paths[val_split:]

        train_generator = MuraGenerator(train_paths, batch_size=16, weights=weights, augment=True if int(args.stage) > 0 else False)
        val_generator = MuraGenerator(val_paths, batch_size=16, weights=weights)

        if not os.path.isdir('logs'):
            os.mkdir('logs')

        logger_path = 'models/' + args.stage + 'model_coco-e{epoch:03d}-l{val_loss:.5f}.hdf5'
        print(logger_path)
        csvlogger = CSVLogger('logs/training' + args.stage + '.log', append=args.resume)
        checkpointer = ModelCheckpoint(logger_path, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                                      patience=1, min_lr=1e-6)

        model.fit_generator(train_generator,
                            callbacks=[csvlogger, checkpointer, reduce_lr],
                            epochs=10,
                            initial_epoch=starting_epoch,
                            validation_data=val_generator)
    elif int(args.stages) is 3:
        img_paths = np.loadtxt(args.test_path, dtype='str')
        img_paths = [str(i) for i in img_paths]

        test_generator = MuraGenerator(img_paths, batch_size=16, weights=[1, 1], augment=False)
        y_pred = model.predict_generator(test_generator)

        sample_weights = []
        y_true = []
        for path in img_paths:
            y_true.append(1 if "positive" in path else 0)
            sample_weights.append(weights[1] if "positive" in path else weights[0])

        print("Scores: ")
        print("\tAccuracy: ", accuracy_score(y_true, y_pred, sample_weight=sample_weights))
        print("\tLoss: ", log_loss(y_true, y_pred, sample_weight=sample_weights))
        print("\tRecall: ", recall_score(y_true, y_pred, sample_weight=sample_weights))
        print("\tPrecision: ", precision_score(y_true, y_pred, sample_weight=sample_weights))
        print("\tAUC: ", roc_auc_score(y_true, y_pred, sample_weight=sample_weights))

    elif int(args.stages) is 4:
        img_paths = np.loadtxt(args.test_path, dtype='str')
        client = args.client.zfill(5)
        img_paths = [str(path) for path in img_paths if client in path]

        images = []
        y_pred = []
        for path in img_paths:
            img = image.load_img(path, color_mode='rgb',
                                 target_size=(320, 320))
            img = transform_image(img, False)
            y_pred.append(model.predict(img))
            images.append(img)

        y_true = 1 if "positive" in img_paths[0] else 0

        print("Results: ")
        print("\ty_true: ", y_true)
        for i in range(len(images)):
            plt.subplot(1, 2, 1)
            plt.imshow(images[i])
            plt.title('Original Image ' + str(y_pred[i]))

            plt.subplot(1, 2, 2)
            plt.imshow(images[i], 'gray', interpolation='none')
            plt.imshow(getCAM(model, images[i]), 'jet', interpolation='none', alpha=0.3)  # TODO: Does this interpolation work
            plt.title('Last Layer')
