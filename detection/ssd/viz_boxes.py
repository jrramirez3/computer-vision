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
from layer_utils import anchor_boxes
import argparse


def show_anchors(image, boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    print(boxes[0][0][0][0])
    print(boxes[0][0][0][1])
    print(boxes[0][0][0][2])
    print(boxes.shape)
    # Create a Rectangle patch
    for i in range(boxes.shape[3]):
        box = boxes[0][0][0][i]
        x = box[0] - (box[2] * 0.5)
        y = box[1] - (box[3] * 0.5)
        w = box[2]
        h = box[3]
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
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
    feature_height = input_shape[1] >> 4
    feature_width = input_shape[2] >> 4 
    input_shape = (1, feature_height, feature_width, input_shape[-1])
    print(input_shape)
    this_scale = 0.2
    aspect_ratios = [0.5, 1.0, 2.0]
    boxes = anchor_boxes(input_shape,
                         img_height,
                         img_width,
                         this_scale,
                         aspect_ratios)


    show_anchors(image, boxes)

