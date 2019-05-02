"""Anchor builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from tensorflow.keras.layers import Layer

from layer_utils import anchor_boxes

class Anchor(Layer):

    def __init__(self,
            image_shape,
            sizes=[1.5, 0.75], 
            aspect_ratios=[1, 2, 0.5],
            **kwargs):

        self.image_shape = image_shape
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.n_boxes = len(aspect_ratios) + len(sizes) - 1
        super(Anchor, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Anchor, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return anchor_boxes(K.int_shape(x),
                            self.image_shape,
                            sizes=self.sizes,
                            aspect_ratios=self.aspect_ratios,
                            x=x)

    def compute_output_shape(self, input_shape):
        batch_size, feature_height, feature_width, feature_channels = input_shape
        return (batch_size, feature_height, feature_width, self.n_boxes, 4)
