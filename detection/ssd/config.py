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
<<<<<<< HEAD
        'epoch_offset': 30,
        'n_classes' : 1,
        'aspect_ratios': [1, 2, 0.5],
        'classes' : ["background", "head"],
=======
        'epoch_offset': 0,
        'aspect_ratios': [1, 2, 0.5],
        'class_thresh' : 0.8,
        'iou_thresh' : 0.2,
        'is_soft_nms' : True,
        'n_classes' : 3,
        'classes' : ["background", "Summit", "Coke", "Pine Juice"],
>>>>>>> 129845f0200b066c3e9de5925dfafb3850a5c2a0
        'prices' : [0.0, 10.0, 40.0, 35.0]
        }
