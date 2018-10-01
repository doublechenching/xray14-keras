# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras import layers
from keras import backend as K


def se_block(x, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.

    Args:
        x: 4d tensor, (batch_size, height, width, channel)
    """
    se_feature = layers.GlobalAveragePooling2D()(x)
    channel = x._keras_shape[-1]
    se_feature = layers.Reshape((1, 1, channel))(se_feature)
    se_feature = layers.Dense(channel // ratio,
                              activation='relu',
                              kernel_initializer='he_normal',
                              use_bias=True,
                              bias_initializer='zeros')(se_feature)
    se_feature = layers.Dense(channel,
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=True,
                              bias_initializer='zeros')(se_feature)
    se_feature = layers.multiply([x, se_feature])

    return se_feature


def channel_attention(input_feature, ratio=8, name=None):
    """similar implementation like Squeeze-and-Excitation(SE) block.

    Args:
        input_feature: 4d tensor, channel last (batch_size, height, width, channel)
        ratio: int, reduce ratio of channel
        name: str, block name
    """
    channel = input_feature._keras_shape[-1]
    shared_layer_one = layers.Dense(channel // ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros',
                                    name=name + '_squeeze')
    shared_layer_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros',
                                    name=name + '_extraction')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid', name=name + 'act')(cbam_feature)
    cbam_feature = layers.multiply([input_feature, cbam_feature])
    return cbam_feature


def spatial_attention(input_feature, kernel_size=7, name=None):
    """2d spatial attention, using single conv layer with big kernel size get
    a one channel saliency map
    x -> [max pool(x) aver pool(x)] -> conv -> multiply(x)
    Args:
        input_feature: 4d tensor, (batch size, height, width, channel)

    """
    cbam_feature = input_feature
    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = layers.concatenate([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(filters=1,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='sigmoid',
                                 kernel_initializer='he_normal',
                                 use_bias=False,
                                 name=name + '_sliency')(concat)

    return layers.multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8, name=None):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.

    Args:
        cbam_feature: input tensor, (batch_size, height, width, channel)
        ratio: int, channel reduce ratio
    """
    cbam_feature = channel_attention(cbam_feature, ratio, name=name + '_channel')
    cbam_feature = spatial_attention(cbam_feature, name=name + '_spatial')

    return cbam_feature


def conv_bn_relu(x, n_filter, kernel_size, name,
                 activation='relu',
                 strides=(1, 1),
                 padding='valid',
                 use_bias=False):
    """naive convolutin block
    x -> Conv ->BN -> Act(optional)
    """
    x = layers.Conv2D(n_filter, kernel_size, 
               strides=strides, padding=padding,
               use_bias=False, name=name)(x)
    x = layers.BatchNormalization(name=name+'_bn')(x)
    if activation:
        x = layers.Activation('relu', name=name+'_act')(x)

    return x


def sepconv_bn_relu(x, n_filter, kernel_size, name,
                 activation='relu',
                 strides=(1, 1),
                 padding='valid',
                 use_bias=False):
    """depthwise separable convolution block
    x => sepconv -> BN -> Act(optional) 
    """
    x = layers.SeparableConv2D(n_filter, kernel_size, 
                        strides=strides, padding=padding,
                        use_bias=False, name=name)(x)
    x = layers.BatchNormalization(name=name+'_bn')(x)
    if activation:
        x = layers.Activation('relu', name=name+'_act')(x)

    return x


def xception_block1(x, n_filter, kernel_size, name,
                   activation='relu'):
    """down_sampling Xception block
    x => sepconv block -> sepconv block -> Maxpooling -> add(Act(x))
    """
    residual = conv_bn_relu(x, n_filter[1], (1, 1), strides=(2, 2), padding='same', 
                            activation=None, name=name+'_res_conv')
    if activation:
        x = layers.Activation('relu', name=name+'_act')(x)
    x = sepconv_bn_relu(x, n_filter[0], kernel_size, padding='same', name=name+'_sepconv1')
    x = sepconv_bn_relu(x, n_filter[1], kernel_size, padding='same', activation=None, name=name+'_sepconv2')
    x = layers.MaxPooling2D(kernel_size, strides=(2, 2), padding='same', name=name+'_pool')(x)
    x = layers.add([x, residual])
    
    return x


def xception_block2(x, n_filter, kernel_size, name,
                   activation='relu'):
    """Xception block
    x => sepconv block -> sepconv block -> sepconv block-> add(Act(x)) =>
    """
    residual = x
    if activation:
        x = layers.Activation('relu', name=name+'_act')(x)   
    x = sepconv_bn_relu(x, n_filter, kernel_size, padding='same', name=name+'_sepconv1')
    x = sepconv_bn_relu(x, n_filter, kernel_size, padding='same', name=name+'_sepconv2')
    x = sepconv_bn_relu(x, n_filter, kernel_size, padding='same', activation=None, name=name+'_sepconv3')
    x = layers.add([x, residual])

    return x
