"""Model builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

import layer_utils
import label_utils
import config

import os
import skimage
import numpy as np
import argparse

from skimage.io import imread
from data_generator import DataGenerator
from model import build_basenetwork, build_ssd4
from label_utils import build_label_dictionary
from viz_boxes import show_boxes

class TinySSD():
    def __init__(self,
                 index=0):

        self.build_model()

    def build_model(self):
        # load dataset path
        csv_path = os.path.join(config.params['data_path'],
                                config.params['train_labels'])

        # build dictionary and key
        self.dictionary, self.classes  = build_label_dictionary(csv_path)
        self.n_classes = len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))

        # load 1st image and build base network
        image_path = os.path.join(config.params['data_path'],
                                  self.keys[0])
        image = skimage.img_as_float(imread(image_path))
        self.input_shape = image.shape
        basenetwork = build_basenetwork(self.input_shape)
        basenetwork.summary()

        # n_anchors = num of anchors per feature point (eg 4)
        ret = build_ssd4(self.input_shape,
                         basenetwork,
                         n_classes=self.n_classes)
        self.n_anchors, self.feature_shape, self.ssd = ret
        self.ssd.summary()
        # print(feature_shape)
        self.train_generator = DataGenerator(dictionary=self.dictionary,
                                             n_classes=self.n_classes,
                                             params=config.params,
                                             input_shape=self.input_shape,
                                             feature_shape=self.feature_shape,
                                             index=0,
                                             n_anchors=self.n_anchors,
                                             batch_size=32,
                                             shuffle=True)


    def classes_loss(self, y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred)


    def offsets_loss(self, y_true, y_pred):
        offset = y_true[..., 0:4]
        mask = y_true[..., 4:8]
        pred = y_pred[..., 0:4]
        offset *= mask
        pred *= mask
        return K.mean(K.square(pred - offset), axis=-1)
        

    def train_model(self):
        optimizer = Adam(lr=1e-3)
        loss = ['categorical_crossentropy', self.offsets_loss]
        self.ssd.compile(optimizer=optimizer, loss=loss)

        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'tinyssd_weights-{epoch:03d}.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     save_weights_only=True)

        callbacks = [checkpoint]
        self.ssd.fit_generator(generator=self.train_generator,
                               use_multiprocessing=True,
                               callbacks=callbacks,
                               epochs=100,
                               workers=16)

    def load_weights(self, weights):
        print("Loading weights : ", weights)
        self.ssd.load_weights(weights)


    def evaluate(self, image_index=0):
        csv_path = os.path.join(config.params['data_path'],
                                config.params['test_labels'])
        self.test_dictionary, _ = build_label_dictionary(csv_path)
        self.test_keys = np.array(list(self.test_dictionary.keys()))

        image_path = os.path.join(config.params['data_path'],
                                  self.keys[image_index])
        image = skimage.img_as_float(imread(image_path))
        image = np.expand_dims(image, axis=0)
        classes, offsets = self.ssd.predict(image)
        image = np.squeeze(image, axis=0)
        classes = np.argmax(classes[0], axis=1)
        offsets = np.squeeze(offsets)
        print(np.unique(classes, return_counts=True))
        show_boxes(image, classes, offsets)

    def test_generator(self):
        x, y = self.train_generator.test(0)
        print(x.shape)
        print(y[0].shape)
        print(y[1].shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Train model"
    parser.add_argument("-t", "--train", action='store_true', help=help_)
    help_ = "Evaluate model"
    parser.add_argument("-e", "--evaluate", default=False, action='store_true', help=help_)
    help_ = "Image index"
    parser.add_argument("--image_index",
                        default=0,
                        type=int,
                        help=help_)

    args = parser.parse_args()

    tinyssd = TinySSD()
    if args.weights:
        tinyssd.load_weights(args.weights)
        if args.evaluate:
            tinyssd.evaluate(args.image_index)
            
            
    if args.train:
        tinyssd.train_model()

    
