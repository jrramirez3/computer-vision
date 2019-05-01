"""Layer utils

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from tensorflow.keras.layers import Layer


def anchor_boxes(feature_shape,
                 image_shape,
                 sizes=[1.0, 0.75], 
                 aspect_ratios=[1, 2, 0.5],
                 is_K_tensor=True):
    
    n_boxes = len(aspect_ratios) + len(sizes) - 1
    image_height, image_width, _ = image_shape
    batch_size, feature_height, feature_width, _ = feature_shape

    grid_width = image_width / feature_width
    grid_height = image_height / feature_height

    wh_list = []
    for ar in aspect_ratios:
        box_height = grid_height * sizes[0] / np.sqrt(ar)
        box_width = grid_width * sizes[0] * np.sqrt(ar)
        wh_list.append((box_width, box_height))
    for size in sizes[1:]:
        box_height = grid_height * size
        box_width = grid_width * size
        wh_list.append((box_width, box_height))

    wh_list = np.array(wh_list)

    start = grid_height * 0.5
    end = (feature_height - 0.5) * grid_height
    cy = np.linspace(start, end, feature_height)

    start = grid_width * 0.5 
    end = (feature_width - 0.5) * grid_width
    cx = np.linspace(start, end, feature_width)

    cx_grid, cy_grid = np.meshgrid(cx, cy)
    # This is necessary for np.tile() to do what we want further down
    cx_grid = np.expand_dims(cx_grid, -1) 
    # This is necessary for np.tile() to do what we want further down
    cy_grid = np.expand_dims(cy_grid, -1)
    # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
    # where the last dimension will contain `(cx, cy, w, h)`
    boxes_tensor = np.zeros((feature_height, feature_width, n_boxes, 4))
    boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
    boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
    boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
    boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h
    # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
    # boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')
    # boxes_tensor = centroid2corners(boxes_tensor)
    # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
    # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
    boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
    if is_K_tensor:
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (feature_shape[0], 1, 1, 1, 1))
    else:
        boxes_tensor = np.tile(boxes_tensor, (feature_shape[0], 1, 1, 1, 1))
    return boxes_tensor

def centroid2corners(boxes_tensor):
    tensor = np.copy(boxes_tensor).astype(np.float)
    tensor[..., 0] = boxes_tensor[..., 0] - boxes_tensor[..., 2] / 2.0 # Set xmin
    tensor[..., 1] = boxes_tensor[..., 1] - boxes_tensor[..., 3] / 2.0 # Set ymin
    tensor[..., 2] = boxes_tensor[..., 0] + boxes_tensor[..., 2] / 2.0 # Set xmax
    tensor[..., 3] = boxes_tensor[..., 1] + boxes_tensor[..., 3] / 2.0 # Set ymax
    return tensor
