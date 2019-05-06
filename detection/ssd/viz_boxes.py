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
import argparse
import os
from random import randint


def box_color(index=None):
    colors = ['b', 'y', 'w', 'r', 'g', 'c', 'm', 'k']
    if index is None:
        return colors[randint(0, len(colors) - 1)]
    return colors[index % len(colors)]

def show_anchors(image, feature_shape, boxes, which_anchors=None, labels=False):
    image_height, image_width, _ = image.shape
    batch_size, feature_height, feature_width, _ = feature_shape

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Show grids
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

    rows = [i for i in range(boxes.shape[1])]
    cols = [i for i in range(boxes.shape[2])]
    for which_anchor in which_anchors:
        i = which_anchor[0]
        j = which_anchor[1]
        if not j in rows:
            continue
        if not i in cols:
            continue
        color = box_color()
        for k in range(boxes.shape[3]):
            # default box format is cx, cy, w, h
            box = boxes[0][j][i][k]
            x = box[0] - (box[2] * 0.5)
            y = box[1] - (box[3] * 0.5)
            w = box[2]
            h = box[3]
            # Rectangle ((xmin, ymin), width, height) 
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='c', facecolor='none')
            ax.add_patch(rect)

    if not labels:
        plt.show()

    return fig, ax


def show_labels(image, labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.imshow(image)
    for label in labels:
        # default label formal is xmin, xmax, ymin, ymax
        w = label[1] - label[0]
        h = label[3] - label[2]
        x = label[0] #+ (w * 0.5)
        y = label[2] #+ (h * 0.5)
        category = int(label[4])
        # Rectangle ((xmin, ymin), width, height) 
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=box_color(category), facecolor='none')
        ax.add_patch(rect)
    plt.show()


def loadcsv(path):
    data = []
    with open(path) as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        # rows = rows[1:]
        for row in rows:
            data.append(row)

    return np.array(data)
    # return np.genfromtxt(path, dtype=[str, np.uint8, np.uint8, np.uint8, np.uint8, np.uint8], delimiter=',')

def dict_label(labels, keys):
    dic = {}
    for key in keys:
        boxes = []
        dic[key] = boxes
        #print(key)
    for label in labels:
        value = label[1:]
        value = value.astype(np.float32)
        key = label[0]
        boxes = dic[key]
        boxes.append(value)
        dic[key] = boxes
        #print(boxes)

    return dic


if __name__ == '__main__':
    data_path = 'dataset/udacity_driving_datasets'
    parser = argparse.ArgumentParser()
    help_ = "Image to visualize"
    parser.add_argument("--image",
                        default = '1479506174991516375.jpg',
                        help=help_)
    help_ = "Receptive field size factor"
    parser.add_argument("--size", default=6, type=int, help=help_)
    help_ = "Show anchors"
    parser.add_argument("--anchors",
                        default=False,
                        action='store_true',
                        help=help_)

    parser.add_argument('--which_anchors',
                        nargs='*',
                        help='<Required> Set flag')

    help_ = "Show labels"
    parser.add_argument("--labels",
                        default=False,
                        action='store_true',
                        help=help_)
    args = parser.parse_args()
    args = parser.parse_args()

    image_path = os.path.join(data_path, args.image)
    image = skimage.img_as_float(imread(image_path))

    which_anchors = None
    ax = None
    if args.which_anchors is not None:
        if len(args.which_anchors) % 2 != 0:
            exit(0)
        which_anchors = np.array(args.which_anchors).astype(np.uint8)
        which_anchors = np.reshape(which_anchors, [-1, 2])
 
        feature_height = image.shape[0] >> args.size
        feature_width = image.shape[1] >> args.size
        feature_shape = (1, feature_height, feature_width, image.shape[-1])
        boxes = anchor_boxes(feature_shape,
                             image.shape,
                             is_K_tensor=False)
        _, ax = show_anchors(image, feature_shape, boxes, which_anchors, args.labels)

    if args.labels:
        csv_file = os.path.join(data_path, 'labels_train.csv')
        labels = loadcsv(csv_file)
        labels = labels[1:]
        keys = np.unique(labels[:,0])
        dic = dict_label(labels, keys)
        print(args.image)
        print(dic[args.image])
        show_labels(image, dic[args.image], ax)
