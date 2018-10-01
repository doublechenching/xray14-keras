#encoding: utf-8
from __future__ import print_function
import os
import matplotlib
matplotlib.use('Agg')
from config import config as cfg
from training import get_number_of_steps
from proc.preproc import (describe_data, load_from_text,
                          split_patients_by_patient_ID,
                          get_class_weight)
from proc.gennerator import (random_image_generator as train_gennerator,
                             ImageGeneratorFromPath)
from proc.image import ImageTransformer
from model import Xception_CBAM
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from training.callback import (MultiGPUCheckpointCallback, HistoryLogger, 
                               MultipleClassAUROC)
from utils import makedir
from keras.utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from metrics import weight_binary_ce


def show_batch_sample(gen, path=None):
    from matplotlib import pyplot as plt
    from skimage.util.montage import montage2d
    batch_x, batch_y = next(gen)
    x = montage2d(np.squeeze(batch_x[:, :, :, 0]))
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(x, cmap='bone')
    plt.axis('off')
    if path:
        plt.savefig(path)
    else:
        plt.show()


def get_callbacks(val_gen, xray14_labels, log_dir, base_model=None):
    auc_cb = MultipleClassAUROC(val_gen, xray14_labels, log_dir,
                                'best_auc_weight.h5', len(val_gen))
    his_cb = HistoryLogger('his', log_dir, plot_item='binary_accuracy')
    weights_path = os.path.join(log_dir, cfg.weights_name)
    if base_model:
        checkpoint = MultiGPUCheckpointCallback(filepath=weights_path, base_model=base_model)
    else:
        checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', 
                                     verbose=1, save_best_only=True, 
                                     mode='min', save_weights_only=True)
    callbacks = [
        TensorBoard(log_dir=log_dir, batch_size=cfg.batch_size),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                          patience=cfg.patience, verbose=1, 
                          mode="min", min_lr=1e-8),
        auc_cb, his_cb, checkpoint
    ]

    return callbacks


def train(exp_name='xray14_wce_06'):
    # train_val_df, test_df, xray14_labels = load_from_text(cfg.data_root)
    train_val_test_df_name = 'train_val_test_df.npy'
    # np.save(train_val_test_df_name, [train_val_df, test_df, xray14_labels])
    train_val_df, test_df, xray14_labels = np.load(train_val_test_df_name)
    train_df, val_df = split_patients_by_patient_ID(train_val_df, 4, cfg.random_seed)
    print('*'*40, 'tain data', '*'*40)
    describe_data(train_df, xray14_labels)
    print('*' * 40, 'val data', '*' * 40)
    describe_data(val_df, xray14_labels)
    train_transformer = ImageTransformer(samplewise_normalization=True, 
                                         rotation_range=10,
                                         width_shift_range=0.05,
                                         height_shift_range=0.1,
                                         shear_range=0.1,
                                         zoom_range=[0.8, 1.2],
                                         horizontal_flip=True)
    val_transformer = ImageTransformer(samplewise_normalization=True)
    train_gen = train_gennerator(train_transformer,
                                 train_val_df,
                                 cfg.input_shape[:-1],
                                 xray14_labels,
                                 batch_size=cfg.batch_size,
                                 color_mode='rgb')
    val_gen = ImageGeneratorFromPath(val_transformer,
                                     test_df['path'],
                                     test_df['xray14_vec'],
                                     shuffle=False,
                                     target_size=cfg.input_shape[:-1],
                                     batch_size=cfg.batch_size,
                                     color_mode='rgb')
    cb_gen = ImageGeneratorFromPath(val_transformer,
                                    test_df['path'],
                                    test_df['xray14_vec'],
                                    shuffle=False,
                                    target_size=cfg.input_shape[:-1],
                                    batch_size=cfg.batch_size,
                                    color_mode='rgb')
    log_path = os.path.join(cfg.log_dir, exp_name)
    makedir(log_path)
    trainable = True
    if cfg.n_gpus > 1:
        with tf.device('/cpu:0'):
            base_model = Xception_CBAM(cfg.input_shape, include_top=True, 
                                       classes=len(xray14_labels),
                                       pretrain_weights='imagenet',
                                       layer_trainable=trainable)
        parallel_model = multi_gpu_model(base_model, gpus=cfg.n_gpus)
        callbacks = get_callbacks(cb_gen, xray14_labels, log_path, base_model=base_model)
        model = parallel_model
    else:
        model = Xception_CBAM(cfg.input_shape, include_top=True, 
                              classes=len(xray14_labels),
                              pretrain_weights='imagenet',
                              layer_trainable=trainable)
        callbacks = get_callbacks(cb_gen, xray14_labels, log_path)
        base_model = model
    class_weights = get_class_weight(train_val_df, xray14_labels)
    print('class weights ---', class_weights)
    model.compile(optimizer=Adam(1e-3), loss=weight_binary_ce(class_weights),
                  metrics=['binary_accuracy', 'mae'])
    train_steps = get_number_of_steps(len(train_val_df), cfg.batch_size)
    val_steps   = get_number_of_steps(len(test_df), cfg.batch_size)
    model.fit_generator(train_gen, epochs=cfg.epochs,
                        steps_per_epoch=train_steps,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        workers=cfg.n_works,
                        max_queue_size=cfg.n_queue,
                        use_multiprocessing=True,
                        validation_steps=val_steps,
                        initial_epoch=0)
    base_model.save_weights(__file__.split('.'[0]) + '.hdf5')


if __name__ == "__main__":
    print(cfg)
    K.set_session(cfg.sess)
    train()