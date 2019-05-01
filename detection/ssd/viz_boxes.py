"""Visualize bounding boxes

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage
from skimage.io import imread
import matplotlib.pyplot as plt
# from layer_utils import anchor_boxes
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Image to visualize"
    parser.add_argument("--image", default='dataset/udacity_driving_datasets/1479506174991516375.jpg', help=help_)
    help_ = "CSV file with labels"
    parser.add_argument("--csv", default='dataset/udacity_driving_datasets/labels_train.csv', help=help_)
    args = parser.parse_args()

    image = skimage.img_as_float(imread(args.image))

    input_shape = (480, 300, 3)
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()


