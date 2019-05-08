"""Labels utils

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import csv
import math
from random import randint


def get_box_color(index=None):
    colors = ['w', 'r', 'b', 'g', 'c', 'y', 'g', 'c', 'm', 'k']
    if index is None:
        return colors[randint(0, len(colors) - 1)]
    return colors[index % len(colors)]


def index2class(index=0):
    classes = ["background", "car", "truck", "pedestrian", "street light"]
    return classes[index]


def load_csv(path):
    data = []
    with open(path) as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            data.append(row)

    return np.array(data)


def get_label_dictionary(labels, keys):
    dictionary = {}
    # boxes = []
    for key in keys:
        dictionary[key] = [] # boxes

    for label in labels:
        value = label[1:]
        value = value.astype(np.float32)
        key = label[0]
        boxes = dictionary[key]
        boxes.append(value)
        dictionary[key] = boxes

    return dictionary
