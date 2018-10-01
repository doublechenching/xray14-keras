#encoding: utf-8
from __future__ import print_function
import os
from training import init_env
import tensorflow as tf

class Struct:
    """pesudo struct
    """
    def __init__(self):
        pass
        
    def __str__(self):
        print("net work config")
        print("*"*80)
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        return '\n'.join(str_list)

config = Struct()
config.gpu = '5'
config.n_gpus = len(config.gpu.split(','))
config.gpu_options = init_env(config.gpu)
config.sess = tf.Session(config=tf.ConfigProto(gpu_options=config.gpu_options, allow_soft_placement=True))
config.data_root = '/home/share/data_repos/chest_xray'
config.input_shape = [256, 256, 3]
config.steps = 'auto'
config.epochs = 100
config.n_works = 15
config.n_queue = 200
config.val_steps = 'auto'
config.batch_size = 32
config.weights_name = 'train_epoch_{epoch:02d}.hdf5'
config.log_dir = './logs'
config.train_val_list_file = os.path.join(config.data_root, 'train_val_list.txt')
config.test_list_file = os.path.join(config.data_root, 'test_list.txt')
config.random_seed = 42
config.patience = 5

if __name__ == "__main__":
    print(config)