"""Model builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import ELU, MaxPooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
# from anchor import Anchor

import layer_utils
import label_utils
import config

import os
import skimage
import numpy as np
import argparse

from skimage.io import imread
from data_generator import DataGenerator

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
               filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               name='conv'+postfix)
    x = BatchNormalization(name="bn"+postfix)(x)
    x = ELU(name='elu'+postfix)(x)
    if use_maxpool:
        x = MaxPooling2D(name='pool'+postfix)(x)
    return x


def build_basenetwork(input_shape,
                      n_layers=1,
                      name='base_network'):

    inputs = Input(shape=input_shape)
    conv1 = conv_layer(inputs,
                       32,
                       kernel_size=5,
                       strides=2,
                       use_maxpool=False,
                       postfix="1")

    conv2 = conv_layer(conv1,
                       64,
                       kernel_size=3,
                       strides=2,
                       use_maxpool=False,
                       postfix="2")

    conv3 = conv_layer(conv2,
                       64,
                       kernel_size=3,
                       strides=2,
                       use_maxpool=False,
                       postfix="3")

    outputs = []
    prev_conv = conv3
    n_filters = 64

    for i in range(n_layers):
        postfix = "_layer" + str(i+1)
        conv = conv_layer(prev_conv,
                          n_filters,
                          kernel_size=3,
                          strides=2,
                          use_maxpool=False,
                          postfix=postfix)
        outputs.append(conv)
        prev_conv = conv
        n_filters *= 2
    
    basenetwork = Model(inputs, outputs, name=name)

    return basenetwork



def build_ssd(input_shape,
              basenetwork,
              n_layers=1,
              n_classes=6):
    # 6 classes = (background, car, truck, 
    # pedestrian, traffic light, street light)
    sizes = layer_utils.anchor_sizes()[0]
    aspect_ratios = layer_utils.anchor_aspect_ratios()

    # number of anchors per feature pt
    n_anchors = len(aspect_ratios) + len(sizes) - 1

    inputs = Input(shape=input_shape)
    base_outputs = basenetwork(inputs)
    print(base_outputs)
    outputs = []
    feature_shapes = []
    out_cls = []
    out_off = []

    for i in range(n_layers):
        conv = base_outputs if n_layers==1 else base_outputs[i]
        name = "cls" + str(i+1)
        classes  = conv2d(conv,
                          n_anchors*n_classes,
                          kernel_size=3,
                          name=name)

        # `offsets`: `(batch, height, width, n_anchors * 4)`
        name = "off" + str(i+1)
        offsets  = conv2d(conv,
                          n_anchors*4,
                          kernel_size=3,
                          name=name)

        shape = np.array(K.int_shape(offsets))[1:]
        feature_shapes.append(shape)

        # reshape the class predictions, yielding 3D tensors of 
        # shape `(batch, height * width * n_anchors, n_classes)`
        # last axis to perform softmax on them
        name = "cls_res" + str(i+1)
        classes = Reshape((-1, n_classes), 
                          name=name)(classes)

        # reshape the offset predictions, yielding 3D tensors of
        # shape `(batch, height * width * n_anchors, 4)`
        # last axis to compute the smooth L1 or L2 loss
        name = "off_res" + str(i+1)
        offsets = Reshape((-1, 4),
                          name=name)(offsets)
        offsets = [offsets, offsets]
        name = "off_cat" + str(i+1)
        offsets = Concatenate(axis=-1,
                              name=name)(offsets)

        out_off.append(offsets)

        name = "cls_out" + str(i+1)
        classes = Activation('softmax',
                             name=name)(classes)

        out_cls.append(classes)

    name = "offsets"
    offsets = Concatenate(axis=1,
                          name=name)(out_off)
    name = "classes"
    classes = Concatenate(axis=1,
                          name=name)(out_cls)

    outputs = [classes, offsets]

    # predictions = [classes, offsets]
    # outputs.append(predictions)

    model = Model(inputs=inputs, outputs=outputs)

    return n_anchors, feature_shapes, model


def build_ssd_orig(input_shape,
              basenetwork,
              n_classes=5):
    sizes = layer_utils.anchor_sizes()
    aspect_ratios = layer_utils.anchor_aspect_ratios()
    # 5 classes = background, car, truck, pedestrian, traffic light
    n_anchors = len(aspect_ratios) + len(sizes) - 1

    inputs = Input(shape=input_shape)
    conv4, conv5, conv6, conv7 = basenetwork(inputs)



    classes4  = conv2d(conv4,
                       n_anchors*n_classes,
                       kernel_size=3,
                       name='classes4')
    classes5  = conv2d(conv5,
                       n_anchors*n_classes,
                       kernel_size=3,
                       name='classes5')
    classes6  = conv2d(conv6,
                       n_anchors*n_classes,
                       kernel_size=3,
                       name='classes6')
    classes7  = conv2d(conv7,
                       n_anchors*n_classes,
                       kernel_size=3,
                       name='classes7')

    # Output shape of `offsets`: `(batch, height, width, n_anchors * 4)`
    offsets4  = conv2d(conv4,
                     n_anchors*4,
                     kernel_size=3,
                     name='offsets4')
    offsets5  = conv2d(conv5,
                     n_anchors*4,
                     kernel_size=3,
                     name='offsets5')
    offsets6  = conv2d(conv6,
                     n_anchors*4,
                     kernel_size=3,
                     name='offsets6')
    offsets7  = conv2d(conv7,
                     n_anchors*4,
                     kernel_size=3,
                     name='offsets7')


    anchors4 = Anchor(input_shape,
                      index=0,
                      name='anchors4')(offsets4)
    anchors5 = Anchor(input_shape,
                      index=1,
                      name='anchors5')(offsets5)
    anchors6 = Anchor(input_shape,
                      index=2,
                      name='anchors6')(offsets6)
    anchors7 = Anchor(input_shape,
                      index=3,
                      name='anchors7')(offsets7)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_anchors, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)

    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_anchors, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    offsets4_reshaped = Reshape((-1, 4), name='offsets4_reshape')(offsets4)
    offsets5_reshaped = Reshape((-1, 4), name='offsets5_reshape')(offsets5)
    offsets6_reshaped = Reshape((-1, 4), name='offsets6_reshape')(offsets6)
    offsets7_reshaped = Reshape((-1, 4), name='offsets7_reshape')(offsets7)


    # Reshape the anchor coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_anchors, 4)`
    anchors4_reshaped = Reshape((-1, 4), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 4), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 4), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 4), name='anchors7_reshape')(anchors7)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_anchors_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped,
                                                                 classes7_reshaped])

    # Output shape of `boxes_concat`: (batch, n_anchors_total, 4)
    offsets_concat = Concatenate(axis=1, name='offsets_concat')([offsets4_reshaped,
                                                             offsets5_reshaped,
                                                             offsets6_reshaped,
                                                             offsets7_reshaped])

    # Output shape of `anchors_concat`: (batch, n_anchors_total, 4)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped,
                                                                 anchors7_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_anchors_total, n_classes + 4 + 8)
    # predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat])
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, offsets_concat, anchors_concat])

    model = Model(inputs=inputs, outputs=predictions)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Image to visualize"
    parser.add_argument("--image",
                        default = '1479506174991516375.jpg',
                        help=help_)
    args = parser.parse_args()

    image_path = os.path.join(config.params['data_path'], args.image)
    image = skimage.img_as_float(imread(image_path))
    input_shape = image.shape
    base = build_basenetwork(input_shape)
    base.summary()

    csv_path = os.path.join(config.params['data_path'],
                            config.params['train_labels'])
    _, classes  = label_utils.build_label_dictionary(csv_path)
    n_classes = len(classes)
    n_anchors, feature_shape, ssd = build_ssd4(input_shape, base, n_classes=n_classes)
    ssd.summary()
    print(feature_shape)
    train_generator = DataGenerator(params=config.params,
                                    input_shape=input_shape,
                                    feature_shape=feature_shape,
                                    index=0,
                                    n_anchors=n_anchors,
                                    batch_size=32,
                                    shuffle=True)
    x, y = train_generator.test(0)
    print(x.shape)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    # plot_model(ssd, to_file="ssd.png", show_shapes=True)
