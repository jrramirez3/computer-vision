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
                 batch_size=32,
                 n_classes=5,
                 shuffle=True):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        csv_path = os.path.join(params['data_path'],
                                params[data_split])
        self.params = params
        self.dictionary = label_utils.build_label_dictionary(csv_path)
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

        for key in keys:
            filename = 
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
