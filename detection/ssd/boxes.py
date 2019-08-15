"""Visualize bounding boxes

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage
import matplotlib.pyplot as plt
import argparse
import os
import layer_utils
import label_utils
import config
import math

from skimage.io import imread
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from layer_utils import anchor_boxes, minmax2centroid, centroid2minmax
from label_utils import index2class, get_box_color


def nms(classes,
        offsets,
        anchors,
        class_thresh=0.9,
        iou_thresh=0.2,
        is_soft=True):
    # get all non-zero (non-background) objects
    objects = np.argmax(classes, axis=1)
    nonbg = np.nonzero(objects)[0]
    #print("Candidate non bg: ", nonbg.size)

    indexes = []
    while True:
        scores = np.zeros((classes.shape[0],))
        scores[nonbg] = np.amax(classes[nonbg], axis=1)
        score_idx = np.argmax(scores, axis=0)
        score_max = scores[score_idx]
        # print(score_max)
        nonbg = nonbg[nonbg != score_idx]
        if score_max < class_thresh:
            if nonbg.size == 0:
                break
            continue
        indexes.append(score_idx)
        score_anc = anchors[score_idx]
        score_off = offsets[score_idx][0:4]
        score_box = score_anc + score_off
        score_box = np.expand_dims(score_box, axis=0)
        nonbg_copy = np.copy(nonbg)
        for idx in nonbg_copy:
            anchor = anchors[idx]
            offset = offsets[idx][0:4]
            box = anchor + offset
            box = np.expand_dims(box, axis=0)
            iou = layer_utils.iou(box, score_box)[0][0]
            if iou >= iou_thresh:
                print(score_idx, "overlaps ", idx, "with iou ", iou)
                if is_soft:
                    iou = iou * iou
                    iou /= 0.5
                    classes[idx] *= math.exp(-iou)
                    print("Scaling...")
                else:
                    nonbg = nonbg[nonbg != idx]
                    print("Removing ...")

        if nonbg.size == 0:
            break


    scores = np.zeros((classes.shape[0],))
    scores[indexes] = np.amax(classes[indexes], axis=1)
    print("Validated non bg: ", len(indexes))
    #if is_soft:
    #    print("Soft NMS")

    return objects, indexes, scores


# image must be normalized (0.0, 1.0)
def show_boxes(image,
               classes,
               offsets,
               feature_shapes,
               show=True,
               normalize=False):

    # generate all anchors per feature map
    anchors = []
    n_layers = len(feature_shapes)
    for index, shape in enumerate(feature_shapes):
        shape = (1, *shape)
        anchor = anchor_boxes(shape,
                              image.shape,
                              index=index)
        anchor = np.reshape(anchor, [-1, 4])
        if index == 0:
            anchors = anchor
        else:
            anchors = np.concatenate((anchors, anchor), axis=0)

    # get all non-zero (non-background) objects
    # objects = np.argmax(classes, axis=1)
    # print(np.unique(objects, return_counts=True))
    # nonbg = np.nonzero(objects)[0]
    if normalize:
        print("Normalize")
        anchors_centroid = minmax2centroid(anchors)
        offsets[:, 0:2] *= 0.1
        offsets[:, 0:2] *= anchors_centroid[:, 2:4]
        offsets[:, 0:2] += anchors_centroid[:, 0:2]
        offsets[:, 2:4] *= 0.2
        offsets[:, 2:4] = np.exp(offsets[:, 2:4])
        offsets[:, 2:4] *= anchors_centroid[:, 2:4]
        offsets = centroid2minmax(offsets)
        # convert fr cx,cy,w,h to real offsets
        offsets[:, 0:4] = offsets[:, 0:4] - anchors

    objects, indexes, scores = nms(classes,
                                   offsets,
                                   anchors,
                                   is_soft=True)

    class_names = []
    rects = []
    if show:
        fig, ax = plt.subplots(1)
        ax.imshow(image)
    for idx in indexes:
        #batch, row, col, box
        anchor = anchors[idx] 
        offset = offsets[idx]
        
        anchor += offset[0:4]
        # default anchor box format is 
        # xmin, xmax, ymin, ymax
        w = anchor[1] - anchor[0]
        h = anchor[3] - anchor[2]
        x = anchor[0]
        y = anchor[2]
        category = int(objects[idx])
        class_name = index2class(category)
        class_name = "%s: %0.2f" % (class_name, scores[idx])
        class_names.append(class_name)
        rect = (x, y, w, h)
        print(class_name, rect)
        rects.append(rect)
        if show:
            color = get_box_color(category)
            rect = Rectangle((x, y),
                             w,
                             h,
                             linewidth=2,
                             edgecolor=color,
                             facecolor='none')
            ax.add_patch(rect)
            bbox = dict(color='none', alpha=1.0)
            ax.text(anchor[0] + 2,
                    anchor[2] - 16,
                    class_name,
                    color=color,
                    fontweight='bold',
                    bbox=bbox,
                    fontsize=8,
                    verticalalignment='top')

    if show:
        plt.show()

    return class_names, rects


def show_anchors(image,
                 feature_shape,
                 anchors,
                 maxiou_indexes=None,
                 maxiou_per_gt=None,
                 labels=None,
                 show_grids=False):
    image_height, image_width, _ = image.shape
    _, feature_height, feature_width, _ = feature_shape

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    if show_grids:
        grid_height = image_height // feature_height
        for i in range(feature_height):
            y = i * grid_height
            line = Line2D([0, image_width], [y, y])
            ax.add_line(line)

        grid_width = image_width // feature_width
        for i in range(feature_width):
            x = i * grid_width
            line = Line2D([x, x], [0, image_height])
            ax.add_line(line)

    # maxiou_indexes is (4, n_gt)
    for index in range(maxiou_indexes.shape[1]):
        i = maxiou_indexes[1][index]
        j = maxiou_indexes[2][index]
        k = maxiou_indexes[3][index]
        # color = label_utils.get_box_color()
        box = anchors[0][i][j][k] #batch, row, col, box
        # default anchor box format is xmin, xmax, ymin, ymax
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = box[0]
        y = box[2]
        # Rectangle ((xmin, ymin), width, height) 
        rect = Rectangle((x, y),
                         w,
                         h,
                         linewidth=2,
                         edgecolor='y',
                         facecolor='none')
        ax.add_patch(rect)

        if maxiou_per_gt is not None and labels is not None:
            # maxiou_per_gt[index] is row w/ max iou
            iou = np.amax(maxiou_per_gt[index])
            #argmax_index = np.argmax(maxiou_per_gt[index])
            #print(maxiou_per_gt[index])
            # offset
            label = labels[index]
            category = int(label[4])
            class_name = index2class(category)
            color = label_utils.get_box_color(category)
            bbox = dict(facecolor=color, color=color, alpha=1.0)
            ax.text(label[0],
                    label[2],
                    class_name,
                    color='w',
                    fontweight='bold',
                    bbox=bbox,
                    fontsize=8,
                    verticalalignment='top')
            dxmin = label[0] - box[0]
            dxmax = label[1] - box[1]
            dymin = label[2] - box[2]
            dymax = label[3] - box[3]
            print(index, ":", "(", class_name, ")", iou, dxmin, dxmax, dymin, dymax)

    if labels is None:
        plt.show()

    return fig, ax
