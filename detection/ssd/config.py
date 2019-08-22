"""Project config"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

params = {
       'data_path' : 'dataset/train_imgs',
       'train_labels' : 'labels_train.csv',
       'test_labels' : 'labels_test.csv',
       'n_classes' : 1,
       'aspect_ratios': [1, 2, 0.5],
       'classes' : ["background", "head"],
       'prices' : [0.0, 10.0, 40.0, 35.0]
       }