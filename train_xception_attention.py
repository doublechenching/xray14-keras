#encoding: utf-8
from __future__ import print_function
import os
from config import config as cfg
from training import get_number_of_steps
from proc.preproc import describe_data, load_from_text, split_patients_by_patient_ID
from proc.gennerator import image_generator_by_diseases, ImageGeneratorFromPath
from proc.image import ImageTransformer
from model import Xception_CBAM
from keras.callbacks import ModelCheckpoint
from utils import makedir
from keras import backend as K
import tensorflow as tf


def train():
    train_val_df, test_df, xray14_labels = load_from_text(cfg.data_root)
    train_df, val_df = split_patients_by_patient_ID(train_val_df, 5)
    print('*'*40, 'tain data', '*'*40)
    describe_data(train_df, xray14_labels)
    print('*' * 40, 'val data', '*' * 40)
    describe_data(val_df, xray14_labels)
    train_transformer = ImageTransformer(samplewise_normalization=True, 
                                         rotation_range=10,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         shear_range=0.1,
                                         zoom_range=[0.7, 1.5],
                                         horizontal_flip=True)
    val_transformer = ImageTransformer(samplewise_normalization=True)
    train_gen = image_generator_by_diseases(train_transformer, train_df, 
                                            cfg.input_shape[:-1], xray14_labels,
                                            batch_size=cfg.batch_size)
    val_gen = ImageGeneratorFromPath(val_transformer, 
                                     val_df['path'], 
                                     val_df['xray14_vec'],
                                     shuffle=False,
                                     target_size=cfg.input_shape[:-1],
                                     batch_size=cfg.batch_size)
    model = Xception_CBAM(cfg.input_shape, include_top=True, classes=len(xray14_labels), pretrain_weights='imagenet')
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['binary_accuracy', 'mae'])
    log_path = os.path.join(cfg.log_dir, 'xcepton_cbam')
    makedir(log_path)
    weights_path = os.path.join(log_path, cfg.weights_name)
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', 
                                 verbose=1, save_best_only=True, 
                                 mode='min', save_weights_only=True)
    callbacks = [checkpoint]
    train_steps = get_number_of_steps(len(train_df), cfg.batch_size)
    val_steps   = get_number_of_steps(len(val_df), cfg.batch_size)
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

