"""Anchor builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from tensorflow.keras.layers import Layer


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
        size = min(self.img_height, self.img_width)
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))

        wh_list = np.array(wh_list)

        batch_size, feature_map_height, feature_map_width, feature_map_channels = K.int_shape(x) #x._keras_shape
        step_height = self.img_height / feature_map_height
        step_width = self.img_width / feature_map_width
        offset_height = 0.5
        offset_width = 0.5

        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down
        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h
        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        # boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')
        tensor = np.copy(boxes_tensor).astype(np.float)
        tensor[..., 0] = boxes_tensor[..., 0] - boxes_tensor[..., 2] / 2.0 # Set xmin
        tensor[..., 1] = boxes_tensor[..., 1] - boxes_tensor[..., 3] / 2.0 # Set ymin
        tensor[..., 2] = boxes_tensor[..., 0] + boxes_tensor[..., 2] / 2.0 # Set xmax
        tensor[..., 3] = boxes_tensor[..., 1] + boxes_tensor[..., 3] / 2.0 # Set ymax
        boxes_tensor = tensor
        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))
        return boxes_tensor


    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 4)
