"""SSD model builder and trainer

Train with 4 layers of feature maps. Pls adjust batch size depending on your GPU memory.
For 1060, -b=1. For V100 32GB, -b=4
python3 ssd.py -l=4 -t -b=4

Train from a previously saved model:
python3 ssd.py -l=4 --weights=saved_models/ResNet56v2_4-layer_weights-200.h5 -t -b=4

Evaluate:
python3 ssd.py -e --weights=saved_models/ResNet56v2_4-layer_weights-200.h5 --image_file=dataset/drinks/0010000.jpg

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

import layer_utils
import label_utils
import config

import os
import skimage
import numpy as np
import argparse

from skimage.io import imread
from data_generator import DataGenerator
from model import build_tinynet, build_ssd
from label_utils import build_label_dictionary
from viz_boxes import show_boxes
from resnet import build_resnet

def lr_scheduler(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 140:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 60:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class SSD():
    def __init__(self,
                 n_layers=4,
                 batch_size=4,
                 epochs=200,
                 workers=16,
                 build_basenet=build_tinynet):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.train_generator = None
        self.test_generator = None
        self.build_model(build_basenet)

    def build_model(self, build_basenet):
        self.build_dictionary()
        # load 1st image and build base network
        image_path = os.path.join(config.params['data_path'],
                                  self.keys[0])
        image = skimage.img_as_float(imread(image_path))
        self.input_shape = image.shape
        self.basenetwork = build_basenet(self.input_shape,
                                         n_layers=self.n_layers)
        self.basenetwork.summary()

        ret = build_ssd(self.input_shape,
                        self.basenetwork,
                        n_layers=self.n_layers,
                        n_classes=self.n_classes)
        # n_anchors = num of anchors per feature point (eg 4)
        # feature_shapes is the feature map shape
        # feature map - basis of class and offset predictions
        self.n_anchors, self.feature_shapes, self.ssd = ret
        self.ssd.summary()


    def build_generator(self):
        # multi-thread train data generator
        self.train_generator = DataGenerator(dictionary=self.dictionary,
                                             n_classes=self.n_classes,
                                             params=config.params,
                                             input_shape=self.input_shape,
                                             feature_shapes=self.feature_shapes,
                                             n_anchors=self.n_anchors,
                                             batch_size=self.batch_size,
                                             shuffle=True)

        return
        # we skip the test data generator since it is time consuming
        # multi-thread test data generator
        self.test_generator = DataGenerator(dictionary=self.test_dictionary,
                                            n_classes=self.n_classes,
                                            params=config.params,
                                            input_shape=self.input_shape,
                                            feature_shapes=self.feature_shapes,
                                            n_anchors=self.n_anchors,
                                            batch_size=self.batch_size,
                                            shuffle=True)


    def build_dictionary(self):
        # load dataset path
        csv_path = os.path.join(config.params['data_path'],
                                config.params['train_labels'])

        # build dictionary and key
        self.dictionary, self.classes  = build_label_dictionary(csv_path)
        self.n_classes = len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))

        return
        csv_path = os.path.join(config.params['data_path'],
                                config.params['test_labels'])
        self.test_dictionary, _ = build_label_dictionary(csv_path)
        self.test_keys = np.array(list(self.test_dictionary.keys()))


    #def classes_loss(self, y_true, y_pred):
    #    return K.categorical_crossentropy(y_true, y_pred)


    def offsets_loss(self, y_true, y_pred):
        # 1st 4 are offsets
        offset = y_true[..., 0:4]
        # last 4 are mask
        mask = y_true[..., 4:8]
        # pred is actually duplicated for alignment
        # either we get the 1st or last 4 offset pred
        # and apply the mask
        pred = y_pred[..., 0:4]
        offset *= mask
        pred *= mask
        # we can use L1 or L2 or soft L1
        return K.mean(K.abs(pred - offset), axis=-1)
        

    def train_model(self):
        if self.train_generator is None:
            self.build_generator()

        optimizer = Adam(lr=1e-3)
        loss = ['categorical_crossentropy', self.offsets_loss]
        self.ssd.compile(optimizer=optimizer, loss=loss)

        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = self.basenetwork.name
        model_name += '_' + str(self.n_layers)
        model_name += '-layer_weights-{epoch:03d}.h5'

        print("Batch size: ", self.batch_size)
        print("Weights filename: ", model_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # prepare callbacks for model saving
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     save_weights_only=True)
        scheduler = LearningRateScheduler(lr_scheduler)

        callbacks = [checkpoint, scheduler]
        self.ssd.fit_generator(generator=self.train_generator,
                               # disable time-consuming validation 
                               # validation_data=self.test_generator,
                               use_multiprocessing=True,
                               callbacks=callbacks,
                               epochs=self.epochs,
                               workers=self.workers)


    def load_weights(self, weights):
        print("Loading weights: ", weights)
        self.ssd.load_weights(weights)


    # evaluate image based on image (np tensor) or filename
    def evaluate(self, image_file=None, image=None):
        show = False
        if image is None:
            #target_file = "%07d" % image_index
            #target_file += ".jpg"
            #image_path = os.path.join(config.params['data_path'], image_file)
            # self.test_keys[image_index])
            image = skimage.img_as_float(imread(image_file))
            show = True

        image = np.expand_dims(image, axis=0)
        classes, offsets = self.ssd.predict(image)
        # print("Classes shape: ", classes.shape)
        # print("Offsets shape: ", offsets.shape)
        image = np.squeeze(image, axis=0)
        # classes = np.argmax(classes[0], axis=1)
        classes = np.squeeze(classes)
        # classes = np.argmax(classes, axis=1)
        offsets = np.squeeze(offsets)
        class_names, rects = show_boxes(image,
                                        classes,
                                        offsets,
                                        self.feature_shapes,
                                        show=show)
        return class_names, rects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Use tinynet as base network"
    parser.add_argument("--tiny",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Num of ssd top feature map layers"
    parser.add_argument("-l",
                        "--layers",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Batch size"
    parser.add_argument("-b",
                        "--batch_size",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Number of workers thread"
    parser.add_argument("--workers",
                        default=8,
                        type=int,
                        help=help_)
    help_ = "Train model"
    parser.add_argument("-t",
                        "--train",
                        action='store_true',
                        help=help_)
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w",
                        "--weights",
                        help=help_)

    help_ = "Evaluate model"
    parser.add_argument("-e",
                        "--evaluate",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Image file for evaluation"
    parser.add_argument("--image_file",
                        default="0010000.jpg",
                        help=help_)


    args = parser.parse_args()

    # build ssd using simple cnn backbone
    if args.tiny:
        ssd = SSD(n_layers=args.layers,
                  batch_size=args.batch_size,
                  workers=args.workers)
    # build ssd using resnet50 backbone
    else:
        ssd = SSD(n_layers=args.layers,
                  build_basenet=build_resnet,
                  batch_size=args.batch_size,
                  workers=args.workers)

    if args.weights:
        ssd.load_weights(args.weights)
        if args.evaluate:
            ssd.evaluate(image_file=args.image_file)
            
    if args.train:
        ssd.train_model()
