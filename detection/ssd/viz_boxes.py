"""Visualize bounding boxes

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage
import csv
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from layer_utils import anchor_boxes
import layer_utils
import argparse
import os
import math
from random import randint


def box_color(index=None):
    colors = ['w', 'r', 'b', 'g', 'c', 'y', 'g', 'c', 'm', 'k']
    if index is None:
        return colors[randint(0, len(colors) - 1)]
    return colors[index % len(colors)]


def class_names(index=0):
    classes = ["background", "car", "truck", "pedestrian", "street light"]
    return classes[index]

def show_anchors(image,
                 feature_shape,
                 boxes,
                 maxiou_indexes=None,
                 maxiou_per_gt=None,
                 labels=None,
                 show_grids=False):
    image_height, image_width, _ = image.shape
    batch_size, feature_height, feature_width, _ = feature_shape

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
        color = box_color()
        box = boxes[0][i][j][k] #batch, row, col, box
        # default box format is xmin, xmax, ymin, ymax
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = box[0]
        y = box[2]
        # Rectangle ((xmin, ymin), width, height) 
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(rect)

        if maxiou_per_gt is not None and labels is not None:
            # maxiou_per_gt[index] is row w/ max iou
            iou = np.amax(maxiou_per_gt[index])
            # offset
            label = labels[index]

            category = int(label[4])
            class_name = class_names(int(label[4]))
            bbox = dict(facecolor=box_color(category), color=box_color(category), alpha=1.0)
            ax.text(label[0], label[2], class_name, color='w', fontweight='bold', bbox=bbox, fontsize=8, verticalalignment='top')
            dxmin = label[0] - box[0]
            dxmax = label[1] - box[1]
            dymin = label[2] - box[2]
            dymax = label[3] - box[3]
            print(index, ":", label[4], iou, dxmin, dxmax, dymin, dymax)

    if labels is None:
        plt.show()

    return fig, ax


def show_labels(image, labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.imshow(image)
    for label in labels:
        # default label format is xmin, xmax, ymin, ymax
        w = label[1] - label[0]
        h = label[3] - label[2]
        x = label[0]
        y = label[2]
        category = int(label[4])
        # Rectangle ((xmin, ymin), width, height) 
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=box_color(category), facecolor='none')
        ax.add_patch(rect)
    plt.show()


def loadcsv(path):
    data = []
    with open(path) as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            data.append(row)

    return np.array(data)


def dict_label(labels, keys):
    dic = {}
    # boxes = []
    for key in keys:
        dic[key] = [] # boxes

    for label in labels:
        value = label[1:]
        value = value.astype(np.float32)
        key = label[0]
        boxes = dic[key]
        boxes.append(value)
        dic[key] = boxes

    return dic


def feature_boxes(image, index):
    d = np.linspace(0.15, 0.8, 6)
    sizes = []
    for i in range(len(d)-1):
        # size = [d[i], math.sqrt(d[i] * d[i + 1])]
        size = [d[i], (d[i] * 0.5)]
        sizes.append(size)
    
    shift = [4, 5, 6, 7, 8] # image div by 2**4 to 2**8
    feature_height = image.shape[0] >> shift[index]
    feature_width = image.shape[1] >> shift[index]
    feature_shape = (1, feature_height, feature_width, image.shape[-1])
    boxes = anchor_boxes(feature_shape,
                         image.shape,
                         sizes=sizes[index],
                         is_K_tensor=False)
    return feature_shape, boxes


if __name__ == '__main__':
    data_path = 'dataset/udacity_driving_datasets'
    parser = argparse.ArgumentParser()
    help_ = "Image to visualize"
    parser.add_argument("--image",
                        default = '1479506174991516375.jpg',
                        help=help_)

    help_ = "Index of receptive field (0 to 3)"
    parser.add_argument("--index",
                        default=0,
                        type=int,
                        help=help_)

    help_ = "Show grids"
    parser.add_argument("--show_grids",
                        default=False,
                        action='store_true',
                        help=help_)

    parser.add_argument('--maxiou_indexes',
                        nargs='*',
                        help='<Required> Set flag')

    help_ = "Show labels"
    parser.add_argument("--labels",
                        default=False,
                        action='store_true',
                        help=help_)
    args = parser.parse_args()

    image_path = os.path.join(data_path, args.image)
    image = skimage.img_as_float(imread(image_path))

    maxiou_indexes = None
    ax = None
    if args.maxiou_indexes is not None:
        if len(args.maxiou_indexes) % 2 != 0:
            exit(0)
        maxiou_indexes = np.array(args.maxiou_indexes).astype(np.uint8)
        maxiou_indexes = np.reshape(maxiou_indexes, [4, -1])
 
        feature_shape, boxes = feature_boxes(image, args.index)
        _, ax = show_anchors(image,
                            feature_shape,
                            boxes,
                            maxiou_indexes=maxiou_indexes,
                            maxiou_per_gt=None,
                            labels=None,
                            show_grids=args.show_grids)
        exit(0)
        #print("Orig boxes shape ", boxes.shape)

    if args.labels:
        csv_file = os.path.join(data_path, 'labels_train.csv')
        labels = loadcsv(csv_file)
        labels = labels[1:]
        keys = np.unique(labels[:,0])
        dic = dict_label(labels, keys)
        labels = dic[args.image]
        # print(labels)

        labels_category = np.array(labels)
        labels = labels_category[:,0:-1]

        feature_shape, boxes = feature_boxes(image, args.index)
        reshaped_boxes = np.reshape(boxes, [-1, 4])
        anchors_array_shape = boxes.shape[0:4]
        print("Labels shape ", labels.shape)
        print("Boxes shape ", boxes.shape)
        print("Anchors array shape ", anchors_array_shape)

        iou = layer_utils.iou(reshaped_boxes, labels)
        print("IOU shape:", iou.shape)
        maxiou_per_gt, maxiou_indexes = layer_utils.maxiou(iou,
                                                           anchors_array_shape)
        _, ax = show_anchors(image,
                             feature_shape,
                             boxes,
                             maxiou_indexes=maxiou_indexes,
                             maxiou_per_gt=maxiou_per_gt,
                             labels=labels_category,
                             show_grids=False)
        show_labels(image, labels_category, ax)
