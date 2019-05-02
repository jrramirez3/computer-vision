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


def show_anchors(image, feature_shape, boxes):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    image_height, image_width, _ = image.shape
    batch_size, feature_height, feature_width, _ = feature_shape

    fig, ax = plt.subplots(1)
    # Display the image
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

    z = 0
    a = boxes.shape[1] * boxes.shape[2] 
    for _ in range(4):
        i = np.random.randint(0, a, 1)[0] % boxes.shape[1]
        j = np.random.randint(0, a, 1)[0] % boxes.shape[2]
        color = colors[z%len(colors)]
        z += 1
        for k in range(boxes.shape[3]):
            box = boxes[0][i][j][k]
            x = box[0] - (box[2] * 0.5)
            y = box[1] - (box[3] * 0.5)
            w = box[2]
            h = box[3]
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
    plt.show()

def loadcsv(path):
    data = []
    with open(path) as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
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
        print(key)
    for label in labels:
        value = label[1:]
        key = label[0]
        boxes = dic[key]
        boxes.append(value)
        dic[key] = boxes
        print(boxes)

    return dic



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Image to visualize"
    parser.add_argument("--image", default='dataset/udacity_driving_datasets/1479506174991516375.jpg', help=help_)
    help_ = "CSV file with labels"
    parser.add_argument("--csv", default='dataset/udacity_driving_datasets/labels_train.csv', help=help_)
    help_ = "Receptive field size factor"
    parser.add_argument("--size", default=6, type=int, help=help_)
    args = parser.parse_args()

    image = skimage.img_as_float(imread(args.image))
    feature_height = image.shape[0] >> args.size
    feature_width = image.shape[1] >> args.size
    feature_shape = (1, feature_height, feature_width, image.shape[-1])
    boxes = anchor_boxes(feature_shape,
                         image.shape,
                         is_K_tensor=False)

    # show_anchors(image, feature_shape, boxes)

    labels = loadcsv(args.csv)
    labels = labels[1:]
    print(labels.shape)
    keys = np.unique(labels[:,0])
    print(labels[1])
    print(keys[1])
    print(keys.shape)
    dic = dict_label(labels, keys)
    print(dic[keys[0]])
    print(dic[keys[1]])
