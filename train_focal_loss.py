#encoding: utf-8
from __future__ import print_function
import os
from config import config as cfg
from training import get_number_of_steps
from proc.preproc import load_from_npy, split_train_val, creat_tain_val_generator, describe_data
from model import Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from utils import makedir
from keras.utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf
from training.callback import MultiGPUCheckpointCallback, lr_schedule
from metrics import focal_loss
from keras.optimizers import Adam


def train():
    # train_val_df, test_df, xray14_labels = load_from_text(cfg.data_root)
    train_val_df, test_df, xray14_labels = load_from_npy()
    train_df, valid_df = split_train_val(train_val_df, ratio=0.25)
    print('*'*40, 'tain data', '*'*40)
    describe_data(train_df, xray14_labels)
    print('*' * 40, 'val data', '*' * 40)
    describe_data(valid_df, xray14_labels)
    train_gen, val_gen = creat_tain_val_generator(train_df, valid_df, cfg.input_shape[:-1], 
                                                  batch_size=cfg.batch_size)
    model = Xception(cfg.input_shape, include_top=True, n_class=len(xray14_labels), pretrain_weights='imagenet')
    model.compile(optimizer=Adam(), loss=[focal_loss()], metrics=['binary_accuracy', 'mae'])
    log_path = os.path.join(cfg.log_dir, 'xray14_focal')
    makedir(log_path)
    weights_path = os.path.join(log_path, cfg.weights_name)
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', 
                                 verbose=1, save_best_only=True, 
                                 mode='min', save_weights_only=True)
    callbacks = [checkpoint, LearningRateScheduler(lr_schedule)]
    train_steps = get_number_of_steps(len(train_df), cfg.batch_size)*5
    val_steps = get_number_of_steps(len(valid_df), cfg.batch_size)*2
    model.fit_generator(train_gen, epochs=cfg.epochs,
                                 steps_per_epoch=train_steps,
                                 callbacks=callbacks,
                                 validation_data=val_gen,
                                 workers=cfg.n_works,
                                 max_queue_size=cfg.n_queue,
                                 use_multiprocessing=True,
                                 validation_steps=val_steps,
                                 initial_epoch=0)

if __name__ == "__main__":
    print(cfg)
    K.set_session(cfg.sess)
    train()

