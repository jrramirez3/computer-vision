"""Project config"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

params = {
<<<<<<< HEAD
       'data_path' : 'dataset/train_imgs',
       'train_labels' : 'labels_train.csv',
       'test_labels' : 'labels_test.csv',
       'n_classes' : 1,
       'aspect_ratios': [1, 2, 0.5],
       'classes' : ["background", "head"],
       'prices' : [0.0, 10.0, 40.0, 35.0]
       }
=======
        'dataset' : 'drinks',
        'data_path' : 'dataset/drinks',
        'train_labels' : 'labels_train.csv',
        'test_labels' : 'labels_test.csv',
        'epoch_offset': 0,
        'aspect_ratios': [1, 2, 0.5],
        'gt_label_iou_thresh' : 0.6,
        'class_thresh' : 0.8,
        'iou_thresh' : 0.2,
        'is_soft_nms' : True,
        'n_classes' : 3,
        'classes' : ["background", "Summit", "Coke", "Pine Juice"],
        'prices' : [0.0, 10.0, 40.0, 35.0]
        }
>>>>>>> 50ecb6b7325a6363b530941709e75a8dfea03e2c
