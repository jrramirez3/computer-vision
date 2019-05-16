"""Layer utils

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical

def anchor_sizes():
    d = np.linspace(0.15, 0.8, 6)
    sizes = []
    for i in range(len(d)-1):
        # size = [d[i], math.sqrt(d[i] * d[i + 1])]
        size = [d[i], (d[i] * 0.5)]
        sizes.append(size)

    return sizes

def anchor_aspect_ratios():
    return [1, 2, 0.5]

def anchor_boxes(feature_shape,
                 image_shape,
                 index=0,
                 x=None,
                 is_K_tensor=True):
    
    sizes = anchor_sizes()[index]
    aspect_ratios = anchor_aspect_ratios()
    print("sizes: ", sizes)
    print("aspect ratios: ", aspect_ratios)
    n_boxes = len(aspect_ratios) + len(sizes) - 1
    image_height, image_width, _ = image_shape
    batch_size, feature_height, feature_width, _ = feature_shape

    norm_width = image_width * sizes[0]
    norm_height = image_height * sizes[0]

    wh_list = []
    for ar in aspect_ratios:
        box_height = norm_height / np.sqrt(ar)
        box_width = norm_width * np.sqrt(ar)
        wh_list.append((box_width, box_height))
    for size in sizes[1:]:
        box_height = image_height * size
        box_width = image_width * size
        wh_list.append((box_width, box_height))

    wh_list = np.array(wh_list)

    grid_width = image_width / feature_width
    grid_height = image_height / feature_height

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
    # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
    # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
    boxes_tensor = centroid2minmax(boxes_tensor)
    boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
    if is_K_tensor:
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))
    else:
        boxes_tensor = np.tile(boxes_tensor, (feature_shape[0], 1, 1, 1, 1))
    return boxes_tensor

def centroid2minmax(boxes_tensor):
    tensor = np.copy(boxes_tensor).astype(np.float)
    tensor[..., 0] = boxes_tensor[..., 0] - boxes_tensor[..., 2] / 2.0 # Set xmin
    tensor[..., 1] = boxes_tensor[..., 0] + boxes_tensor[..., 2] / 2.0 # Set xmax
    tensor[..., 2] = boxes_tensor[..., 1] - boxes_tensor[..., 3] / 2.0 # Set ymin
    tensor[..., 3] = boxes_tensor[..., 1] + boxes_tensor[..., 3] / 2.0 # Set ymax
    return tensor

def intersection(boxes1, boxes2):
    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    xmin = 0
    xmax = 1
    ymin = 2
    ymax = 3

    boxes1_min = np.expand_dims(boxes1[:, [xmin, ymin]], axis=1)
    boxes1_min = np.tile(boxes1_min, reps=(1, n, 1))
    boxes2_min = np.expand_dims(boxes2[:, [xmin, ymin]], axis=0)
    boxes2_min = np.tile(boxes2_min, reps=(m, 1, 1))
    min_xy = np.maximum(boxes1_min, boxes2_min)

    boxes1_max = np.expand_dims(boxes1[:, [xmax, ymax]], axis=1)
    boxes1_max = np.tile(boxes1_max, reps=(1, n, 1))
    boxes2_max = np.expand_dims(boxes2[:, [xmax, ymax]], axis=0)
    boxes2_max = np.tile(boxes2_max, reps=(m, 1, 1))
    max_xy = np.minimum(boxes1_max, boxes2_max)

    side_lengths = np.maximum(0, max_xy - min_xy)

    intersection_areas = side_lengths[:, :, 0] * side_lengths[:, :, 1]
    return intersection_areas


def union(boxes1, boxes2, intersection_areas):
    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    xmin = 0
    xmax = 1
    ymin = 2
    ymax = 3

    areas = (boxes1[:, xmax] - boxes1[:, xmin]) * (boxes1[:, ymax] - boxes1[:, ymin])
    boxes1_areas = np.tile(np.expand_dims(areas, axis=1), reps=(1,n))
    areas = (boxes2[:,xmax] - boxes2[:,xmin]) * (boxes2[:,ymax] - boxes2[:,ymin])
    boxes2_areas = np.tile(np.expand_dims(areas, axis=0), reps=(m,1))

    union_areas = boxes1_areas + boxes2_areas - intersection_areas
    return union_areas


def iou(boxes1, boxes2):
    intersection_areas = intersection(boxes1, boxes2)
    union_areas = union(boxes1, boxes2, intersection_areas)
    return intersection_areas / union_areas


def maxiou(iou, anchors_array_shape, n_classes=6, anchors=None, labels=None):
    iou_mask = np.zeros(iou.shape)
    maxiou_per_gt = np.argmax(iou, axis=0)
    print("IOU[0]: ", iou[maxiou_per_gt[0]])
    iou_rows = np.argmax(iou[maxiou_per_gt], axis=1)
    #maxiou = np.append(maxiou_per_gt, iou_rows, axis=1)
    print("IOU Rows shape:", iou_rows.shape)
    print("MaxIOU: ", maxiou_per_gt)
    print("MaxIOU shape: ", maxiou_per_gt.shape)
    a = np.reshape(maxiou_per_gt, (maxiou_per_gt.shape[0], 1))
    b = np.reshape(iou_rows, (iou_rows.shape[0], 1))
    maxiou = np.append(a, b, axis=1)
    print("maxiou shape: ", maxiou.shape)
    # for i in range(maxiou.shape[0]):
    #    iou_mask[maxiou[i][0], maxiou[i][1]]  = 1.0
    iou_mask[maxiou[:,0], maxiou[:,1]]  = 1.0
    print("IOU Mask[0]: ", iou_mask[maxiou_per_gt[0], iou_rows[0]])
    print("IOU[0]: ", iou[maxiou_per_gt[0], iou_rows[0]])
    print("IOU Mask uniques:", np.unique(iou_mask))
    print("IOU Mask Shape: ", iou_mask.shape)
    unique, counts = np.unique(iou_mask, return_counts=True)
    d = dict(zip(unique, counts))
    print("dict of counts: ", d)
    masked_ious = np.multiply(iou, iou_mask)
    print("Masked IOU:", np.unique(masked_ious))
    # maxiou = np.reshape(maxiou, [-1, 2])
    print(maxiou_per_gt)
    print(iou_rows)
    print(maxiou)
   
    # mask generation
    gt_mask = np.zeros((iou.shape[0], 4))
    gt_mask[maxiou_per_gt] = 1.0
    print("Mask uniques:", np.unique(gt_mask, return_counts=True))
    print("Mask Shape: ", gt_mask.shape)
    print("Mask[0]: ", gt_mask[maxiou_per_gt[0]])

    # class generation
    gt_class = np.zeros((iou.shape[0], n_classes))
    gt_class[:, 0] = 1
    gt_class[maxiou_per_gt, 0] = 0
    maxiou_col = np.reshape(maxiou_per_gt, (maxiou_per_gt.shape[0], 1))
    label_col = np.reshape(labels[:,4], (labels.shape[0], 1)).astype(int)
    row_col = np.append(maxiou_col, label_col, axis=1)
    gt_class[row_col[:,0], row_col[:,1]]  = 1.0
    print("Class uniques:", np.unique(gt_class, return_counts=True))
    print("Class Shape: ", gt_class.shape)
    print("Class[0]: ", gt_class[0])
    for i in range(labels.shape[0]):
        print(gt_class[maxiou_per_gt[i]])


    # offset generation
    gt_offset = np.zeros((iou.shape[0], 4))
    anchors = np.reshape(anchors, [-1, 4])
    offsets = labels[0:4] - anchors[maxiou_per_gt]
    gt_offset[maxiou_per_gt] = offsets
    print("Offset Shape: ", gt_offset.shape)
    print("Offset[0]: ", gt_offset[0])
    for i in range(labels.shape[0]):
        print(gt_offset[maxiou_per_gt[i]])

    
    maxiou_indexes = np.array(np.unravel_index(maxiou_per_gt, anchors_array_shape))
    maxiou_per_gt = iou[maxiou_per_gt]
    print("MaxIOU GT shape: ", maxiou_per_gt.shape)
    print("MaxIOU indexes shape: ", maxiou_indexes.shape)
    return maxiou_per_gt, maxiou_indexes
