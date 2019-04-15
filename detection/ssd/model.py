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

def conv_layer(inputs,
               name_postfix,
               filters=16,
               kernel_size=3,
               strides=1,
               activation=None):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer='he_normal',
                  # kernel_regularizer=l2(l2_reg),
                  name="conv"+name_postfix,
                  use_maxpool=True,
                  padding='same')

    x = inputs
    x = conv(x)
    x = BatchNormalization(name="bn"+name_postfix)(x)
    x = ELU(name='elu'+name_postfix)(x)
    if use_maxpool:
        x = MaxPooling2D(name='pool'+name_postfix)(x)
    return x

def build_basenetwork(input_shape,
                      output_shape=None,
                      name=None):

    channels = int(output_shape[-1])

    inputs = Input(shape=input_shape)
    conv1 = conv_layer(inputs,
                       32,
                       kernel_size=5,
                       name_postfix="1")

    conv2 = conv_layer(conv1,
                       48,
                       kernel_size=3,
                       name_postfix="2")

    conv3 = conv_layer(conv2,
                       64,
                       kernel_size=3,
                       name_postfix="3")

    conv4 = conv_layer(conv3,
                       64,
                       kernel_size=3,
                       name_postfix="4")
    
    conv5 = conv_layer(conv4,
                       48,
                       kernel_size=3,
                       name_postfix="5")
    
    conv6 = conv_layer(conv5,
                       48,
                       kernel_size=3,
                       name_postfix="6")
    
    conv7 = conv_layer(conv6,
                       48,
                       kernel_size=3,
                       name_postfix="7",
                       use_maxpool=False)


    classes4 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes4')(conv4)
        classes5 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes5')(conv5)
            classes6 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes6')(conv6)
                classes7 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes7')(conv7)
                    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
                        boxes4 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(conv4)
                            boxes5 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(conv5)
                                boxes6 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes6')(conv6)
                                    boxes7 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes7')(conv7)

    basenetwork = Model(inputs, outputs, name=name)

    return basenetwork


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Build model"
    args = parser.parse_args()
    input_shape = (256, 256, 1)
    output_shape = (256, 256, 1)
    base = build_basenetwork(input_shape, output_shape)
    base.summary()
