"""Project config

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

params = {
        'dataset' : 'drinks',
        'data_path' : 'dataset/drinks',
        'train_labels' : 'labels_train.csv',
        'test_labels' : 'labels_test.csv',
        'epoch_offset': 0,
        'aspect_ratios': [1, 2, 0.5],
        'class_thresh' : 0.9,
        'iou_thresh' : 0.2,
        'is_soft_nms' : True,
        'n_classes' : 3,
        'classes' : ["background", "Summit", "Coke", "Pine Juice"],
        'prices' : [0.0, 10.0, 40.0, 35.0]
        }
