"""Project config

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

params = {
        'data_path' : 'dataset/dataSyn',
        'train_labels' : 'labels_train.csv',
        'test_labels' : 'labels_test.csv',
        'n_classes' : 2,
        'aspect_ratios': [1, 1, 1],
        'classes' : ["background", "Head"],
        # 'prices' : [0.0, 10.0, 40.0, 35.0]
        }
