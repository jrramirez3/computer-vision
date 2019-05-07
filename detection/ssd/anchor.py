"""Anchor builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from tensorflow.keras.layers import Layer

from layer_utils import anchor_boxes
import layer_utils

class Anchor(Layer):

    def __init__(self,
            image_shape,
            index=0,
            **kwargs):

        self.image_shape = image_shape
        self.index = index
        self.sizes = layer_utils.anchor_sizes()[index]
        self.aspect_ratios = layer_utils.anchor_aspect_ratios()
        self.n_boxes = len(self.aspect_ratios) + len(self.sizes) - 1
        super(Anchor, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Anchor, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return anchor_boxes(K.int_shape(x),
                            self.image_shape,
                            index=self.index,
                            x=x)

    def compute_output_shape(self, input_shape):
        batch_size, feature_height, feature_width, feature_channels = input_shape
        return (batch_size, feature_height, feature_width, self.n_boxes, 4)
