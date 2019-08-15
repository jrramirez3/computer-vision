"""Label utility functions

Main use: labeling, dictionary of colors,
label retrieval, loading label csv file,
drawing label on an image

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import csv
import config
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from random import randint

# retrieve plt-compatible color string based on object index
def get_box_color(index=None):
    colors = ['w', 'r', 'b', 'g', 'c', 'm', 'y', 'g', 'c', 'm', 'k']
    if index is None:
        return colors[randint(0, len(colors) - 1)]
    return colors[index % len(colors)]

# retrieve rgb color based on object index
def get_box_rgbcolor(index=None):
    colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 128, 0)]
    if index is None:
        return colors[randint(0, len(colors) - 1)]
    return colors[index % len(colors)]

# convert index (int) to class name (string)
def index2class(index=0):
    classes = config.params['classes']
    return classes[index]

# convert class name (string) to index (int)
def class2index(class_="background"):
    classes = config.params['classes']
    return classes.index(class_)

# load a csv file into an np array
def load_csv(path):
    data = []
    with open(path) as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            data.append(row)

    return np.array(data)

# associate key (filename) to value (box coords, class)
def get_label_dictionary(labels, keys):
    dictionary = {}
    # boxes = []
    for key in keys:
        dictionary[key] = [] # boxes

    for label in labels:
        value = label[1:]

        if value[0]==value[1]:
            continue
        if value[2]==value[3]:
            continue

        if label[-1]==0:
            print("No object labelled as bg:", label[0])
            continue
        value = value.astype(np.float32)
        key = label[0]
        boxes = dictionary[key]
        boxes.append(value)
        dictionary[key] = boxes

    return dictionary


# build a dict with key=filename, value=[box coords, class]
def build_label_dictionary(csv_path):
    labels = load_csv(csv_path)
    # skip the 1st line header
    labels = labels[1:]
    keys = np.unique(labels[:,0])
    dictionary = get_label_dictionary(labels, keys)
    classes = np.unique(labels[:,-1]).astype(int).tolist()
    classes.insert(0, 0)
    print("Num of unique classes: ", classes)
    return dictionary, classes


# draw bounding box on an object given box coords (labels[1:5])
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
        color = get_box_color(category)
        # Rectangle ((xmin, ymin), width, height) 
        rect = Rectangle((x, y),
                         w,
                         h,
                         linewidth=2,
                         edgecolor=color,
                         facecolor='none')
        ax.add_patch(rect)
    plt.show()
