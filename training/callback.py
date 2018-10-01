#encoding: utf-8
from __future__ import print_function
from keras.callbacks import Callback
from keras import backend as K
import warnings
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import os
from matplotlib import pyplot as plt

class MultiGPUCheckpointCallback(Callback):
    """save weights when you use keras multi gpu model in data parallel mode

    # Args:
        filepath: formated string, weights path
        base_model: model instance in cpu
    """
    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model

    # Args:

    # Examples:
    ```python
    cb = MultipleClassAUROC(val_gen, xray_labels, 
                            '/home/project/best_auc_score.h5',
                            len(val_gen))
    ```
    """
    def __init__(self, val_gen, class_names, log_dir, weights_name, steps=0):
        super(MultipleClassAUROC, self).__init__()
        self.val_gen = val_gen
        self.class_names = class_names
        self.weights_name = weights_name
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, 'auc_log.csv')
        self.epochs_cls_auc = []
        self.epochs_aver_auc = []
        if not steps or steps > len(val_gen):
            self.steps = len(val_gen)
        else:
            self.steps = steps

    def predict(self):
        y_pred = []
        y_true = []
        for batch_id in range(self.steps):
            batch_x, batch_y = self.val_gen[batch_id]
            pred = self.model.predict(batch_x, batch_size=self.val_gen.batch_size)
            score = np.mean(np.round(pred[:]) == batch_y[:])
            print('predicting batch ', batch_id + 1, ', total', self.steps, '---- accuracy score: ', score)
            y_pred.append(pred)
            y_true.append(batch_y)
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        return y_true, y_pred

    def plot_class_roc(self, y_true, y_pred, epoch, path):
        class_auroc = []
        fig, axes = plt.subplots(1, 1, figsize=(9, 9))
        for i in range(len(self.class_names)):
            try:
                fpr, tpr, thresholds = roc_curve(y_true[:, i].astype(int), y_pred[:, i])
                score = auc(fpr, tpr)
                axes.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (self.class_names[i], score))
            except ValueError:
                score = 0
            class_auroc.append(score)
            print("epoch ", epoch, "---class ", self.class_names[i], " score --- ", score)
        axes.legend()
        axes.set_xlabel('False Positive Rate')
        axes.set_ylabel('True Positive Rate')
        name = os.path.join(path, 'epoch'+str(epoch)+'_roc.png')
        fig.savefig(name)
        plt.close(fig)
        return class_auroc

    def plot_aver_roc(self, y_true, y_pred, epoch, path):
        y_true = y_true.flatten().astype(int)
        y_pred = y_pred.flatten()
        fig, axes = plt.subplots(1, 1, figsize=(9, 9))
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            auc_score = auc(fpr, tpr)
            axes.plot(fpr, tpr, label='%s (AUC:%0.2f)' % ('average', auc_score))
        except ValueError:
            auc_score = 0
        axes.legend()
        axes.set_xlabel('False Positive Rate')
        axes.set_ylabel('True Positive Rate')
        name = os.path.join(path, 'epoch'+str(epoch)+'_aver_roc.png')
        fig.savefig(name)
        print('epoch ', epoch, ' average auc is ', auc_score)
        plt.close(fig)
        return auc_score


    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.
        """
        print("\n", "*"*30)
        lr = float(K.eval(self.model.optimizer.lr))
        print("current learning rate: ", lr)
        y_true, y_pred = self.predict()
        class_auc = self.plot_class_roc(y_true, y_pred, epoch, self.log_dir)
        aver_auc = self.plot_aver_roc(y_true, y_pred, epoch, self.log_dir)
        if self.epochs_aver_auc:
            auc_max = np.amax(self.epochs_aver_auc)
        else:
            auc_max = 0

        if aver_auc >= auc_max:
            name = 'epoch' + str(epoch) + '_' + self.weights_name
            weights_path = os.path.join(self.log_dir, name)
            self.model.save_weights(weights_path)

        with open(self.log_file, mode='a+') as f:
            str_list = ['%.3f' % item for item in class_auc]
            str_list.append('%.3f' % aver_auc)
            print(','.join(str_list), file=f)

        self.epochs_cls_auc.append(class_auc)
        self.epochs_aver_auc.append(aver_auc)

    def on_train_begin(self, logs={}):
        """
        训练开始, 重载父类函数
        """
        with open(self.log_file, mode='a+') as f:
            print(','.join(self.class_names), ',aver auc', file=f)

class HistoryLogger(Callback):
    """save trining history and plot training epoch loss
    
    # Args:
        name_prefix: name prefix of file which is used for saving traing history
        file_path: str, log dir
        plot_item: str, keras logs key, such as 'val loss' or metric name

    # Examples:
    ```python
    log_dir = '/home/project/abus'
    his_cb = HistoryLogger('his', log_dir, plot_item='binary_accuracy')
    ```
    """

    def __init__(self, name_prefix='abus_train', 
                 file_path='./logs', mode='a+', 
                 plot_item='loss'):
        self.name_prefix = name_prefix
        file_name = name_prefix + '_history.txt'
        file_name = os.path.join(file_path, file_name)
        self.file = open(file_name, mode)
        self.path = file_path
        self.plot_item = plot_item
        super(HistoryLogger, self).__init__()


    def on_train_begin(self, logs={}):
        """
        训练开始, 重载父类函数
        """
        print("开始训练", file=self.file)
        self.file.flush()
        self.epoch_loss = []


    def on_batch_begin(self, batch, logs={}):
        pass


    def on_batch_end(self, batch, logs={}):
        """
        Args:
            batch: int
            logs:  dict
        """
        log_line = ['%-10s - %-10.5f' % item for item in sorted(logs.items())]
        log_line.insert(0, 'epoch--- %-3d'% self.cur_epoch)
        log_line = '\t'.join(log_line)
        print(log_line, file=self.file)
        self.losses.append(logs)
        self.file.flush()

    def on_epoch_begin(self, epoch, logs={}):
        self.cur_epoch = epoch
        self.losses = []


    def on_epoch_end(self, epoch, logs={}):
        print('*' * 80, file=self.file)
        log_line = ['%-10s - %-10.5f' % item for item in logs.items()]
        log_line = '\t'.join(log_line) + '\n'
        print(log_line, file=self.file)
        print('*' * 80, file=self.file)
        self.file.flush()
        # 画出loss下降曲线
        loss = np.zeros((len(self.losses)))
        for i, item  in enumerate(self.losses):
            output_loss = item.get(self.plot_item)
            loss[i] = output_loss
        plt.figure("epoch-" + str(epoch))
        plt.plot(loss)
        plt.gca().set_xlabel('batch')
        plt.gca().set_ylabel('loss')
        plt.gca().set_title('epoch-{}'.format(epoch))
        save_path = os.path.join(self.path, self.name_prefix + '_epoch_{}.jpg'.format(epoch))
        plt.savefig(save_path)
        # 记录val_loss
        self.epoch_loss.append(logs.get('loss'))

    def on_train_end(self, logs={}):
        plt.figure("epoch loss")
        plt.plot(np.array(self.epoch_loss))
        plt.gca().set_xlabel('epoch')
        plt.gca().set_ylabel('loss')
        plt.gca().set_title('epoch loss')
        save_path = os.path.join(self.path, './epoch_loss.jpg')
        plt.savefig(save_path)
        self.file.close()


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 30, 60, 90, 120 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 120:
        lr *= 0.5e-3
    elif epoch > 90:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr