"""Data generator

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras
import layer_utils
import label_utils
import os

class DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 params={},
                 data_split='train_labels',
                 input_shape=(300, 480, 3),
                 feature_shape=(300, 480, 3),
                 n_anchors=0,
                 batch_size=32,
                 shuffle=True):
        self.params = params
        self.input_shape = input_shape
        self.n_anchors = n_anchors
        self.batch_size = batch_size
        self.shuffle = shuffle
        csv_path = os.path.join(params['data_path'],
                                params[data_split])
        self.dictionary, classes = label_utils.build_label_dictionary(csv_path)
        self.n_classes = len(classes)
        self.keys = np.array(list(self.dictionary.keys()))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dictionary) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        keys = self.keys[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        x, y = self.__data_generation(keys)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.keys)

    def __data_generation(self, keys):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        data_path = params['data_path']
        # Initialization
        x = np.empty((self.batch_size, *self.input_shape))
        gt_class = np.empty((self.batch_size, self.n_anchors, self.n_classes))
        gt_offset = np.empty((self.batch_size, self.n_anchors, 4))
        gt_mask = np.empty((self.batch_size, self.n_anchors, 4))

        for i, key in enumerate(keys):
            image_path = os.path.join(data_path, key)
            image = skimage.img_as_float(imread(image_path))
            x[i] = image
            feature_shape, anchors = feature_boxes(image, self.index)
            labels = self.dictionary[key]
            labels = np.array(labels)
            boxes = labels[:,0:-1]
            iou = layer_utils.iou(anchors, boxes)
            gt_class[i], gt_offset[i], gt_mask[i] = get_gt_data(iou,
                                                                n_classes=self.n_classes,
                                                                anchors=anchors,
                                                                labels=labels)
            

        return y, keras.utils.to_categorical(y, num_classes=self.n_classes)
