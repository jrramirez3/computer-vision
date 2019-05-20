"""Model builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import ELU, MaxPooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import plot_model
from keras import backend as K
from anchor import Anchor

import layer_utils
import label_utils
import config

import os
import skimage
import numpy as np
import argparse
from skimage.io import imread
from data_generator import DataGenerator
import config
from model import build_basenetwork, build_ssd4
from label_utils import build_label_dictionary

class TinySSD():
    def __init__(self,
                 index=0):

        self.build_model()

    def build_model(self):

        csv_path = os.path.join(config.params['data_path'],
                                config.params['train_labels'])
        self.dictionary, self.classes  = build_label_dictionary(csv_path)
        self.n_classes = len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))

        image_path = os.path.join(config.params['data_path'],
                                  self.keys[0])
        image = skimage.img_as_float(imread(image_path))
        input_shape = image.shape
        base = build_basenetwork(input_shape)
        base.summary()

        # n_anchors = num of anchors per feature point (eg 4)
        n_anchors, feature_shape, ssd = build_ssd4(input_shape,
                                                   base,
                                                   n_classes=self.n_classes)
        ssd.summary()
        print(feature_shape)
        self.train_generator = DataGenerator(params=config.params,
                                             input_shape=input_shape,
                                             feature_shape=feature_shape,
                                             index=0,
                                             n_anchors=n_anchors,
                                             batch_size=32,
                                             shuffle=True)

    def test_generator(self):
        x, y = self.train_generator.test(0)
        print(x.shape)
        print(y[0].shape)
        print(y[1].shape)
        print(y[2].shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Image to visualize"
    parser.add_argument("--image",
                        default = '1479506174991516375.jpg',
                        help=help_)
    args = parser.parse_args()

    tinyssd = TinySSD()
    tinyssd.test_generator()
