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


def show_anchors(image, input_shape, boxes):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # Create figure and axes
    height = image.shape[0]
    width = image.shape[1]
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    print(boxes[0][0][0][0])
    print(boxes[0][0][0][1])
    print(boxes[0][0][0][2])
    print(boxes.shape)
    # Show grids
    delta_y = height // input_shape[1]
    for i in range(input_shape[1]):
        y = i * delta_y
        line = Line2D([0, width], [y, y])
        ax.add_line(line)

    delta_x = width // input_shape[2]
    for i in range(input_shape[2]):
        x = i * delta_x
        line = Line2D([x, x], [0, height])
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

    img_height = image.shape[0]
    img_width = image.shape[1]
    input_shape = np.expand_dims(image, axis=0).shape
    feature_height = input_shape[1] >> 6
    feature_width = input_shape[2] >> 6 
    input_shape = (1, feature_height, feature_width, input_shape[-1])
    print(input_shape)
    this_scale = 0.3
    aspect_ratios = [0.5, 1.0, 2.0]
    boxes = anchor_boxes(input_shape,
                         img_height,
                         img_width,
                         this_scale,
                         aspect_ratios,
                         is_K_tensor=False)


    show_anchors(image, input_shape, boxes)

