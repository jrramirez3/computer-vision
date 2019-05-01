"""Visualize bounding boxes

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage
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
    print(boxes[0][0][0][0])
    print(boxes[0][0][0][1])
    print(boxes[0][0][0][2])
    print(boxes.shape)
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
            #rect = Rectangle((200, 100),100,30, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Image to visualize"
    parser.add_argument("--image", default='dataset/udacity_driving_datasets/1479506174991516375.jpg', help=help_)
    help_ = "CSV file with labels"
    parser.add_argument("--csv", default='dataset/udacity_driving_datasets/labels_train.csv', help=help_)
    args = parser.parse_args()

    image = skimage.img_as_float(imread(args.image))
    feature_height = image.shape[0] >> 6
    feature_width = image.shape[1] >> 6 
    feature_shape = (1, feature_height, feature_width, image.shape[-1])
    boxes = anchor_boxes(feature_shape,
                         image.shape,
                         is_K_tensor=False)

    show_anchors(image, feature_shape, boxes)

