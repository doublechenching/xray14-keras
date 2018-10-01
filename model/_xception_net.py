#encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
from keras import layers
from keras import backend as K
from keras import Model
from keras import utils as keras_utils
from ._blocks import (cbam_block, sepconv_bn_relu, xception_block1,
                      xception_block2, conv_bn_relu)

def load_imagenet_weights(model, include_top):
    TF_WEIGHTS_PATH = (
        'https://github.com/fchollet/deep-learning-models/'
        'releases/download/v0.4/'
        'xception_weights_tf_dim_ordering_tf_kernels.h5')
    TF_WEIGHTS_PATH_NO_TOP = (
        'https://github.com/fchollet/deep-learning-models/'
        'releases/download/v0.4/'
        'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if include_top:
        weights_path = keras_utils.get_file(
            'xception_weights_tf_dim_ordering_tf_kernels.h5',
            TF_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
        
    else:
        weights_path = keras_utils.get_file(
            'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='b0042744bf5b25fce3cb969f33bebb97')
    print("pretain from imagenet----", weights_path)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)


def Xception(input_shape=None,
             include_top=True,
             pooling='avg',
             n_class=14,
             pretrain_weights='imagenet'):
    """Instantiates the Xception architecture.
    
    Note that the default input image size for this model is 299x299.

    Args:
        input_shape: list or tuple, (height, width, channel)
        include_top: whether to include the fully-connected
            layer at the top of the network.
        pretrain_weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        n_class: int, num of class
        pooling: string, option of ['max', 'avg']
    Returns:

    Raise:

    Examples:

    """
    inputs = layers.Input(shape=input_shape)
    x = conv_bn_relu(inputs, 32, (3, 3), strides=(2, 2), name='block1_conv1')
    x = conv_bn_relu(x, 64, (3, 3), name='block1_conv2')

    x = xception_block1(x, [128, 128], (3, 3), activation=None, name='block2')
    x = xception_block1(x, [256, 256], (3, 3), name='block3')
    x = xception_block1(x, [728, 728], (3, 3), name='block4')

    for i in range(8):
        prefix = 'block' + str(i + 5)
        x = xception_block2(x, 728, (3, 3), name=prefix)
    # bottle neck
    x = xception_block1(x, [728, 1024], (3, 3), name='block13')
    x = sepconv_bn_relu(x, 1536, (3, 3), name='block14_conv1')
    x = sepconv_bn_relu(x, 2048, (3, 3), name='block14_conv2')
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(n_class, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    # Create model
    model = Model(inputs, x, name='xception')

    # Load weights
    if pretrain_weights == 'imagenet':
        load_imagenet_weights(model, include_top)
    elif os.path.exists(pretrain_weights):
        print("pretain from imagenet----", pretrain_weights)
        model.load_weights(pretrain_weights)
    else:
        print("weight path does not exist!")
    return model


if __name__ == "__main__":
    model = Xception([512, 512, 3], n_class=1000, pretrain_weights='imagenet')
    model.summary()
    keras_utils.plot_model(model, to_file='model.png', show_shapes=True)

