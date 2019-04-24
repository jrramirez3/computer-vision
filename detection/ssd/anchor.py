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
            img_width,
            img_height,
            this_scale,
            aspect_ratios=[0.5, 1.0, 2.0],
            **kwargs):

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.aspect_ratios = aspect_ratios
        self.n_boxes = len(aspect_ratios)
        super(Anchor, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Anchor, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        boxes_tensor = anchor_boxes(x,
                                    self.img_height,
                                    self.img_width,
                                    self.this_scale,
                                    self.aspect_ratios)
        return boxes_tensor


    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 4)
