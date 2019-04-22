import argparse
import os
import random

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras import initializers
from keras.applications import DenseNet169
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.preprocessing import image
from keras.utils import Sequence
# from keras_applications.densenet import preprocess_input
from skimage import transform
from sklearn.metrics import roc_auc_score, precision_score, recall_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd


def preprocess_input(img, data_format):
    img /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]

    img[..., 0] /= std[0]
    img[..., 1] /= std[1]
    img[..., 2] /= std[2]
    return img


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


def get_cam(model, img, predicted_class):
    output = model.output

    densenet_layer = model.get_layer('densenet169')
    final_conv_layer = densenet_layer.get_layer('relu')

    grads = K.gradients(model.output,
                        model.get_layer('densenet169').layers[-1].output)[0]

    print(grads)

    pooled = K.mean(grads, axis=(0, 1, 2))

    print(pooled)
    iterate = K.function([model.input], [pooled, final_conv_layer.output[0]])

    pooled_val, final_conv_val = iterate(img)

    depth = final_conv_layer.shape[-1]

    for i in range(depth):
        final_conv_layer[:, :, i] *= pooled_val[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


class MuraGenerator(Sequence):
    def __init__(self, paths_studies, batch_size, weights, augment=False):
        self.bs = batch_size
        self.paths_studies = paths_studies
        self.augment = augment
        self.weights = weights

    def __len__(self):
        return len(self.paths_studies) // self.bs

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        w_batch = []

        paths = self.paths_studies[idx * self.bs: (idx + 1) * self.bs]
        for path in paths:
            section = path[0][16:23]
            img_paths = glob(str(path[0]) + '*')
            for img_path in img_paths:
                img = image.load_img(img_path, color_mode='rgb',
                                     target_size=(320, 320))

                img = image.img_to_array(img)

                img = transform_image(img, self.augment)

                x_batch.append(img)
                y_batch.append(int(path[1]))
                w_batch.append(self.weights[section][int(path[1])])

        return [np.asarray(x_batch), np.asarray(y_batch), np.asarray(w_batch)]


def generate_model(stage):
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

    adam = optimizers.Adam(lr=0.5e-3)
    sgd = optimizers.SGD(lr=0.5e-3, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  metrics=['accuracy'],
                  loss='binary_crossentropy')

    if args.resume is True or args.stage >= 2:
        model.load_weights(args.model_path, by_name=True)
        print("Path: ", args.model_path)

    if stage is 2:
        for layer in densenet.layers:
            if 'conv5' in layer.name:
                layer.trainable = True
        model.compile(optimizer=sgd,
                      metrics=['accuracy'],
                      loss='binary_crossentropy')

    return model


if __name__ == "__main__":
    to_print=True
    print("JC")
    parser = argparse.ArgumentParser(description="MURA image classification")
    parser.add_argument('-r', '--resume', action='store_true', default='False',
                        help='Resume training from last saved model')
    parser.add_argument('-s', '--stage', default=0, type=int,
                        help='Set stage of training: '
                             '0-train only dense layer.'
                             '1-train only dense layer with image augmentation.'
                             '2-train dense with augmentation and last conv block.'
                             '3-testing, report all metric of the test data.'
                             '4-evaluate a single client, indicated with -c, plot image and CAM.')
    parser.add_argument('--train_path', default='./MURA-v1.1/train_labeled_studies.csv',
                        help='Path to train_labeled_studies.csv')
    parser.add_argument('--train_images', default='./MURA-v1.1/train_image_paths.csv',
                        help='Path to train_image_paths.csv')

    parser.add_argument('--test_path', default='./MURA-v1.1/valid_labeled_studies.csv',
                        help='Path to valid_labeled_studies.csv')
    parser.add_argument('--test_images', default='./MURA-v1.1/valid_image_paths.csv',
                        help='Path to valid_image_paths.csv')
    parser.add_argument('--model_path',
                        help='Path to a model to resume or proceed with transfer learning')
    parser.add_argument('-c', '--client', default=0, type=int,
                        help='Client to evaluate')

    parser.add_argument('--section', default='XR_WRIS',
                        help='XR_SHOU, XR_HUME, XR_FORE, XR_HAND, XR_ELBO, XR_FING, XR_WRIS')


    args = parser.parse_args()

    starting_epoch = 0

    model = generate_model(int(args.stage))

    if args.resume is True:
        starting_epoch = int(args.model_path[25:28])
        print("starting epoch: ", starting_epoch)

    studies_path = np.asarray(pd.read_csv(args.train_path, delimiter=',', header=None))

    weights = {
        "XR_SHOU": [0, 0],
        "XR_HUME": [0, 0],
        "XR_FORE": [0, 0],
        "XR_HAND": [0, 0],
        "XR_ELBO": [0, 0],
        "XR_FING": [0, 0],
        "XR_WRIS": [0, 0]
    }

    paths_imgs = np.loadtxt(args.train_images, dtype='str')
    for path in paths_imgs:
        section = path[16:23]
        if "positive" in path:
            weights[section][1] += 1
        elif "negative" in path:
            weights[section][0] += 1

    for section in weights:
        weights[section] = weights[section]/np.sum(weights[section])

    print(weights)

    if args.stage < 3:

        train_paths, val_paths = train_test_split(studies_path)

        train_generator = MuraGenerator(train_paths, batch_size=8, weights=weights,
                                        augment=True if args.stage > 0 else False)
        val_generator = MuraGenerator(val_paths, batch_size=8, weights=weights)

        if not os.path.isdir('logs'):
            os.mkdir('logs')
        if not os.path.isdir('models'):
            os.mkdir('models')

        checkpoint_path = 'models/stage-{}-'.format(args.stage) + '-model-e{epoch:03d}.hdf5'
        csvlogger = CSVLogger(
            'logs/stage-{}.log'.format(args.stage), append=args.resume)
        checkpointer = ModelCheckpoint(checkpoint_path, save_best_only=False,
                                       save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                                      verbose=1, patience=1, min_lr=1e-7)

        model.fit_generator(train_generator,
                            callbacks=[csvlogger, checkpointer],
                            epochs=10,
                            initial_epoch=starting_epoch,
                            validation_data=val_generator)

    elif args.stage == 3:
        studies_path = np.asarray(pd.read_csv(args.test_path, delimiter=',', header=None))
        if args.section is not None:
            studies_path = [path for path in studies_path if args.section in path]

        y_pred = []
        y_true = []
        y_pred_bin = []
        sample_w = []
        for study in studies_path:
            section = study[0][16:23]
            img_paths = glob(str(study[0]) + '*')
            images = []
            for img_path in img_paths:
                img = image.load_img(img_path, color_mode='rgb',
                                     target_size=(320, 320))

                img = image.img_to_array(img)

                img = transform_image(img, False)
                images.append(img)

            images = np.asarray(images)
            results = model.predict_on_batch(images)
            y_pred.append(np.mean(results))

            y_true.append(int(study[1]))
            sample_w.append(weights[section][int(study[1])])
        print("len y_pred: ", len(y_pred))
        print("y_pred[0]: ", y_pred[0])
        y_pred_bin = [0 if i <= 0.5 else 1 for i in y_pred]
        ones = 0
        zeroes = 0
        for pred in y_pred_bin:
            if int(pred) is 1:
                ones += 1
            if int(pred) is 0:
                zeroes += 1
        print("Ones: ", ones)
        print("Zeroes: ", zeroes)

        print("Scores: ")
        print("Number of studies: ", len(y_true))
        print("\tLoss: ", log_loss(y_true, y_pred))
        print("\tLoss (weights): ", log_loss(y_true, y_pred, sample_weight=sample_w))
        print("\tAccuracy: ", accuracy_score(y_true, y_pred_bin))
        print("\tAccuracy (weights): ", accuracy_score(y_true, y_pred_bin, sample_weight=sample_w))
        print("\tRecall: ", recall_score(y_true, y_pred_bin))
        print("\tRecall (weights): ", recall_score(y_true, y_pred_bin, sample_weight=sample_w))
        print("\tPrecision: ", precision_score(y_true, y_pred_bin))
        print("\tPrecision (weights): ", precision_score(y_true, y_pred_bin, sample_weight=sample_w))
        print("\tAUC: ", roc_auc_score(y_true, y_pred))
        print("\tAUC (weights): ", roc_auc_score(y_true, y_pred, sample_weight=sample_w))

    elif args.stage == 4:
        img_paths = np.loadtxt(args.test_path, dtype='str')
        client = args.client.zfill(5)
        img_paths = [str(path) for path in img_paths if str(client) in path]
        if len(img_paths) == 0:
            print('patient not found on validation set')
            exit()

        images = []
        y_pred = []
        for path in img_paths:
            img = image.load_img(path, color_mode='rgb',
                                 target_size=(320, 320))
            img = image.img_to_array(img)
            img = np.expand_dims(transform_image(img, False), 0)

            y_pred.append(model.predict(img))
            images.append(img)

        y_true = 1 if "positive" in img_paths[0] else 0
        # print(y_pred)
        # y_pred = np.mean(y_pred) # Ensemble the results

        print("Results: ")
        print("\ty_true: ", y_true)
        print("\ty_pred: ", np.mean(y_pred))

        for i in range(len(images)):
            predicted = 0 if np.squeeze(y_pred[i]) < weights[0] else 1

            plt.subplot(1, 2, 1)
            plt.imshow(images[i].squeeze())
            plt.title('Original Image ' + str(y_pred[i]))

            plt.subplot(1, 2, 2)
            plt.imshow(images[i].squeeze(), 'gray')
            plt.imshow(get_cam(model, images[i], predicted), 'jet', alpha=0.3)
            plt.title('Last Layer')

        plt.savefig('logs/map.png')