"""Model builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import BatchNormalization
from keras.layers import ELU, MaxPooling2D
from keras.models import Model
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.utils import plot_model

import argparse

def conv2d(inputs,
           filters=32,
           kernel_size=3,
           strides=1,
           name=None):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer='he_normal',
                  # kernel_regularizer=l2(l2_reg),
                  name=name,
                  padding='same')

    return conv(inputs)


def conv_layer(inputs,
               filters=32,
               kernel_size=3,
               strides=1,
               use_maxpool=True,
               postfix=None,
               activation=None):

    x = conv2d(inputs,
               kernel_size=kernel_size,
               strides=strides,
               name='conv'+postfix)
    x = BatchNormalization(name="bn"+postfix)(x)
    x = ELU(name='elu'+postfix)(x)
    if use_maxpool:
        x = MaxPooling2D(name='pool'+postfix)(x)
    return x

def build_basenetwork(input_shape,
                      output_shape=None,
                      name='base_network'):

    channels = int(output_shape[-1])

    inputs = Input(shape=input_shape)
    conv1 = conv_layer(inputs,
                       32,
                       kernel_size=5,
                       postfix="1")

    conv2 = conv_layer(conv1,
                       48,
                       kernel_size=3,
                       postfix="2")

    conv3 = conv_layer(conv2,
                       64,
                       kernel_size=3,
                       postfix="3")

    conv4 = conv_layer(conv3,
                       64,
                       kernel_size=3,
                       postfix="4")
    
    conv5 = conv_layer(conv4,
                       48,
                       kernel_size=3,
                       postfix="5")
    
    conv6 = conv_layer(conv5,
                       48,
                       kernel_size=3,
                       postfix="6")
    
    conv7 = conv_layer(conv6,
                       48,
                       kernel_size=3,
                       postfix="7",
                       use_maxpool=False)

    basenetwork = Model(inputs, [conv4, conv5, conv6, conv7], name=name)

    return basenetwork


def build_ssd(inputs,
              basenetwork):
              

    conv4, conv5, conv6, conv7 = basenetwork(innputs)

    classes4  = conv2d(conv4,
                       n_boxes[0] * n_classes,
                       kernel_size=3,
                       name='classes4')
    classes5  = conv2d(conv5,
                       n_boxes[1] * n_classes,
                       kernel_size=3,
                       name='classes5')
    classes6  = conv2d(conv6,
                       n_boxes[2] * n_classes,
                       kernel_size=3,
                       name='classes6')
    classes7  = conv2d(conv7,
                       n_boxes[3] * n_classes,
                       kernel_size=3,
                       name='classes7')

    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes4  = conv2d(conv4,
                     n_boxes[0] * 4,
                     kernel_size=3,
                     name='boxes4')
    boxes5  = conv2d(conv5,
                     n_boxes[1] * 4,
                     kernel_size=3,
                     name='boxes5')
    boxes6  = conv2d(conv6,
                     n_boxes[2] * 4,
                     kernel_size=3,
                     name='boxes6')
    boxes7  = conv2d(conv7,
                     n_boxes[3] * 4,
                     kernel_size=3,
                     name='boxes7')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Build model"
    args = parser.parse_args()
    input_shape = (480, 300, 3)
    output_shape = (480, 300, 3)
    base = build_basenetwork(input_shape, output_shape)
    base.summary()
