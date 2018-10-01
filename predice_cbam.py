#encoding: utf-8
from __future__ import print_function
import numpy as np
import os
from config import config as cfg
from proc.gennerator import ImageGeneratorFromPath
from proc.preproc import split_patients_by_patient_ID
from proc.image import ImageTransformer
from model import Xception_CBAM as Xception
from utils import makedir
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


def predict(gen, n_samples, weights_path, classes, batch_size=32):
    model = Xception(cfg.input_shape, include_top=True,
                     pretrain_weights=weights_path,
                     classes=classes,
                     layer_trainable=True)
    model.load_weights(weights_path)
    model.trainable = False
    pred_Y = []
    test_Y = []
    for batch_id in range(len(gen)):
        batch_x, batch_y = gen[batch_id]
        pred = model.predict(batch_x, batch_size=batch_size)
        score = np.mean(np.round(pred[:]) == batch_y[:])
        print('predicting batch ', batch_id + 1, ', total', int(n_samples/batch_size), '---- accuracy score: ', score)
        pred_Y.append(pred)
        test_Y.append(batch_y)
    pred_Y = np.concatenate(pred_Y, axis=0)
    test_Y = np.concatenate(test_Y, axis=0)
    print(pred_Y.shape)
    return pred_Y, test_Y


def eval_roc(pred_Y, test_Y, all_labels):
    # ROC
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    for (idx, c_label) in enumerate(all_labels):
        fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), pred_Y[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    fig.savefig('cbam_test.png')

    for c_label, p_count, t_count in zip(all_labels, 100*np.mean(pred_Y, 0), 100*np.mean(test_Y, 0)):
        print('%s: Dx: %2.2f%%, PDx: %2.2f%%' % (c_label, t_count, p_count))


if __name__ == "__main__":
    train_val_test_df_name = 'train_val_test_df.npy'
    train_val_df, test_df, xray14_labels = np.load(train_val_test_df_name)

    # train_val_df, test_df, xray14_labels = load_from_text(cfg.data_root)
    train_df, val_df = split_patients_by_patient_ID(train_val_df, 4, cfg.random_seed)
    test_transformer = ImageTransformer(samplewise_normalization=True)

    test_df = test_df

    test_gen = ImageGeneratorFromPath(test_transformer,
                                      test_df['path'],
                                      test_df['xray14_vec'],
                                      shuffle=False,
                                      target_size=cfg.input_shape[:-1],
                                      batch_size=cfg.batch_size,
                                      color_mode='rgb')
    log_path = os.path.join(cfg.log_dir, 'xray14_wce_05')
    weights_path = os.path.join(log_path, 'train_epoch_33.hdf5')
    print("weights path ----", weights_path)
    pred_Y, test_Y = predict(test_gen, len(test_df), weights_path,
                             len(xray14_labels), batch_size=cfg.batch_size)
    eval_roc(pred_Y, test_Y, xray14_labels)
