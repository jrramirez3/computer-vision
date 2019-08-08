"""Project config

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

params = {
        'dataset' : 'heads',
        'data_path' : 'dataset/heads',
        'train_labels' : 'labels_train.csv',
        'test_labels' : 'labels_test.csv',
        'epoch_offset': 30,
        'n_classes' : 1,
        'aspect_ratios': [1, 2, 0.5],
        'classes' : ["background", "head"],
        'prices' : [0.0, 10.0, 40.0, 35.0]
        }
