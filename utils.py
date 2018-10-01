#encoding: utf-8
from __future__ import print_function
from matplotlib import pyplot as plt
import os

def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print('exist folder---', path)


def history_plot(history, save_path=''):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
