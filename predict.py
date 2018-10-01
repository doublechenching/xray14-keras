#encoding: utf-8
from __future__ import print_function
import numpy as np
import os
from config import config as cfg
from proc.preproc import load_from_text, split_train_val, flow_from_dataframe, load_from_npy
from keras.preprocessing.image import ImageDataGenerator
# from model import Xception_CBAM as Xception
from model import Xception
from utils import makedir
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
cfg.batch_size = 32
def load_data():
    train_val_df, test_df, xray14_labels = load_from_npy()
    test_idg = ImageDataGenerator(samplewise_center=True,
                                  samplewise_std_normalization=True)
    test_gen = flow_from_dataframe(test_idg,
                                   test_df,
                                   path_col='path',
                                   y_col='xray14_vec',
                                   target_size=cfg.input_shape[:-1],
                                   color_mode='grayscale',
                                   batch_size=cfg.batch_size)
    return test_gen, test_df, xray14_labels


def predict(gen, n_samples, weights_path, classes, batch_size=32):
    model = Xception(cfg.input_shape, include_top=True,
                     pretrain_weights=weights_path, classes=classes)
    pred_Y = []
    test_Y = []
    for batch_id in range(int(n_samples/batch_size)):
        batch_x, batch_y = next(gen)
        pred = model.predict(batch_x, batch_size=batch_size)
        score = np.mean(np.round(pred[:]) == batch_y[:])
        print('predicting batch ', batch_id + 1, ', total', int(n_samples/batch_size), '---- accuracy score: ', score)
        pred_Y.append(pred)
        test_Y.append(batch_y)
    pred_Y = np.concatenate(pred_Y, axis=0)
    test_Y = np.concatenate(test_Y, axis=0)
    return pred_Y, test_Y


def eval_roc(pred_Y, test_Y, all_labels):
    # ROC
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    for (idx, c_label) in enumerate(all_labels):
        fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), pred_Y[:, idx])
        print('threshold is ', thresholds)
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')

    fig.savefig('trained_ce200.png')

    for c_label, p_count, t_count in zip(all_labels, 100*np.mean(pred_Y, 0), 100*np.mean(test_Y, 0)):
        print('%s: 预测阳性比例: %2.2f%%, 实际阳性比例: %2.2f%%' % (c_label, t_count, p_count))

# def show_samples(test_Y, all_labels):
#     sickest_idx = np.argsort(np.sum(test_Y, 1) < 1)
#     fig, m_axs = plt.subplots(4, 2, figsize=(16, 32))
#     for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
#         c_ax.imshow(test_X[idx, :, :, 0], cmap='bone')
#         stat_str = [n_class[:6] for n_class, n_score in zip(all_labels,
#                                                             test_Y[idx])
#                     if n_score > 0.5]
#         pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100) for n_class, n_score, p_score in zip(all_labels,
#                                                                                                    test_Y[idx], pred_Y[idx])
#                     if (n_score > 0.5) or (p_score > 0.5)]
#         c_ax.set_title('Dx: '+', '.join(stat_str) +
#                        '\nPDx: '+', '.join(pred_str))
#         c_ax.axis('off')
#     fig.savefig('trained_img_predictions.png')


if __name__ == "__main__":
    test_gen, test_df, xray14_labels= load_data()
    log_path = os.path.join(cfg.log_dir, 'xcepton_cbam')
    weights_path = os.path.join(log_path, 'train_epoch_02.hdf5')
    print("weights path ----", weights_path)
    pred_Y, test_Y = predict(test_gen, len(test_df), weights_path, len(xray14_labels), batch_size=cfg.batch_size)
    eval_roc(pred_Y, test_Y, xray14_labels)