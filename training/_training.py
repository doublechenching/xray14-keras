#encoding: utf-8
from __future__ import print_function
from keras import backend as K
import tensorflow as tf
import os
from keras.utils import multi_gpu_model
import numpy as np

def init_env(cuda_vis_dev='0'):
    """init trainging environment

    # Args:
        cuda_vis_dev: str, visiable gpu devices
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_vis_dev
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.97, allow_growth=True)
    return gpu_options

def get_number_of_steps(n_samples, batch_size):
    """get keras training or validation steps
    
    # Args
        n_samples: int, number of samples in dataset
        batch_size：int, batch size

    # Returns
        kreas trian steps
    """
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:  # 取余
        return n_samples // batch_size
    else:
        return n_samples // batch_size + 1


def train_on_batch(model, data_generator, n_epoch=3, steps=10000,
                   weights_prefix="./abus_weights_{0}.h5"):
    """训练

    基于论文CASED，实现课程训练

    Args:
        imglist: volume列表
        labellist: 标签列表
        patchdim: 一个cube size的元祖
        epoch_num: 训练的轮数
        batch_size: batch大小，当值为1的时候为在线学习
        epoch: 一轮的迭代次数，一般为数据集的大小
        model: 模型网络
        saveiter: 每隔多少次迭代保存一次权重
        modelname: 权重文件的前缀字符串

    Returns:
        权重文件名字符串
    """
    for epoch in range(n_epoch):
        for step in range(1, steps + 1):
            # gen data
            batch_x, batch_y = next(data_generator)
            print('training ', str(epoch), ' th epoch---', 'step ', str(step))
            output = model.train_on_batch(batch_x, batch_y)
            loss_list = ['loss%2d\t%.3f' % loss for loss in enumerate(output)]
            print('\t'.join(loss_list))
        model.save_weights(weights_prefix)


def train_on_data_parallel(CPU_model, optimizer, loss_funcs, metrics,
                           generator, steps, epochs, callback, 
                           n_works=5, val_gen=None, val_steps=None,
                           gpus=1, init_epoch=0):
    """note that model must instantiated in cpu

    Example:   
    ```python
        with tf.device('/cpu:0'):
            model = get_model(config)
        model = train_on_data_parallel(model, optimizer, loss, metrics,
                data_gen, steps, epochs)
        model.save_weights('myweights.h5')
    ```
    """
    parallel_model = multi_gpu_model(CPU_model, gpus=gpus)
    parallel_model.compile(loss=loss_funcs,
                           metrics=metrics,
                           optimizer=optimizer)
    parallel_model.fit_generator(generator, steps_per_epoch=steps,
                                 epochs=epochs, callbacks=callback,
                                 validation_data=val_gen,
                                 workers=n_works,
                                 use_multiprocessing=True,
                                 validation_steps=val_steps,
                                 initial_epoch=init_epoch)
    return CPU_model


if __name__ == "__main__":
    n = get_number_of_steps(100, 2)
    print(n)
