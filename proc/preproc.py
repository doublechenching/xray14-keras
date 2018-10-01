#encoding: utf-8
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import glob
from itertools import chain
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def creat_generator(train_df, valid_df, idg, output_size=(256, 256), batch_size=16):
    """creat image generator for trainning and validation

    # Args:
        train_df: dataframe, include training image path and label
        valid_df: dataframe, include training image path and label
        idg: keras ImageGenerator instance
        output_size: tuple or list, original image will be resized to this size
    """
    train_gen = flow_from_dataframe(idg[0], train_df,
                                    path_col='path',
                                    y_col='xray14_vec',
                                    target_size=output_size,
                                    color_mode='grayscale',
                                    batch_size=batch_size)
    valid_gen = flow_from_dataframe(idg[1], valid_df,
                                    path_col='path',
                                    y_col='xray14_vec',
                                    target_size=output_size,
                                    color_mode='grayscale',
                                    batch_size=batch_size)
    return train_gen, valid_gen


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    """get image generator flowing batch image from dataframe
    
    # Args:
        img_data_gen: keras ImageGenerator instance
        in_df: input dataframe
        path_col: str, path column
        y_col: sre, label column
    """
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('---Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                              class_mode='sparse',
                                              **dflow_args)
    # 固定filenames和classes
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    # 样本大小
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ''  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

    return df_gen


def describe_data(df, labels):
    """analyse class number of dataset

    # Args:
        df: dataframe
        labels: xray14 or xray8 class name
    """
    print('There are total ', len(df), ' images.')
    label_cnt = [[c_label, int(df[c_label].sum())] for c_label in labels]
    label_cnt = sorted(label_cnt, key=lambda x:x[1], reverse=True)
    print('\n'.join(['%-20s---%s' % (c_label[0], c_label[1]) for c_label in label_cnt]))


def load_from_text(data_root):
    """split dataframe using offical train_val and test spliting text

    # Args:
        data_root: xray14 dataset path
    """
    train_val_txt_path = os.path.join(data_root, 'train_val_list.txt')
    test_txt_path = os.path.join(data_root, 'test_list.txt')
    data_entry_path = os.path.join(data_root, 'Data_Entry_2017.csv')
    xray_df = pd.read_csv(data_entry_path)
    image_paths = glob.glob(os.path.join(data_root, 'images', 'images*', '*.png'))
    all_image_paths = {os.path.basename(x): x for x in image_paths}
    print("*"*80)
    assert len(all_image_paths) == xray_df.shape[0], "the num of data entries must be equal with origin images"
    xray_df['path'] = xray_df['Image Index'].map(all_image_paths.get)
    
    xray_df['Finding Labels'] = xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    # 查找疾病的种类
    all_labels = np.unique(list(chain(*xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    xray14_labels = [x for x in all_labels if len(x) > 0]
    train_val_name = []
    test_name = []
    with open(train_val_txt_path, mode='r') as f:
        for name in f.readlines():
            train_val_name.append(name[:-1])

    with open(test_txt_path, mode='r') as f:
        for name in f.readlines():
            test_name.append(name[:-1])
    
    train_val_df = xray_df.loc[xray_df['Image Index'].isin(train_val_name)].copy()
    test_df = xray_df.loc[xray_df['Image Index'].isin(test_name)].copy()

    for c_label in xray14_labels:
        xray_df[c_label] = xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    for c_label in xray14_labels:
        train_val_df[c_label] = train_val_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
        
    for c_label in xray14_labels:
        test_df[c_label] = test_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    print('There are total ', len(train_val_df), ' train and val data')
    print('There are total ', len(test_df), ' test data')
    print("*"*40, 'all data', "*"*40)
    describe_data(xray_df, xray14_labels)
    print("*"*40, 'train val data', "*"*40)
    describe_data(train_val_df, xray14_labels)
    print("*"*40, 'test data', "*"*40)
    describe_data(test_df, xray14_labels)
    train_val_df['xray14_vec'] = train_val_df.apply(lambda x: [ x[xray14_labels].values ], 1).map(lambda x: x[0])
    test_df['xray14_vec'] = test_df.apply(lambda x: [ x[xray14_labels].values ], 1).map(lambda x: x[0])
    
    return train_val_df, test_df, xray14_labels


def split_train_val(train_val_df, ratio=0.25, seed=42):
    """random split train_val dataframe, note that this method is not patient-wise spliting
    
    # Args:
        train_val_df: dataframe, training and validation dataframe
        ratio: float, spliting ratio
        seed: int, set a random seed, get same reslut every time
    """
    train_df, valid_df = train_test_split(train_val_df,
                                          test_size=ratio,
                                          random_state=seed,
                                          stratify=train_val_df['Finding Labels'].map(lambda x: x[:4]))
    return train_df, valid_df


def creat_tain_val_generator(train_df, valid_df, output_size,
                             batch_size=32,
                             filp=True, shift_range=[0.05, 0.1], 
                             shear_range=0.1, zoom_range=[0.7, 1.5], 
                             rotation_range=5):
    """creat image generator for trainning and validation
    """
    train_idg = ImageDataGenerator(samplewise_center=True,
                                   samplewise_std_normalization=True,
                                   horizontal_flip=filp,
                                   vertical_flip=False,
                                   height_shift_range=shift_range[0],
                                   width_shift_range=shift_range[1],
                                   rotation_range=rotation_range,
                                   shear_range=shear_range,
                                   fill_mode='constant',
                                   cval=0,
                                   zoom_range=zoom_range
                                  )
    val_idg = ImageDataGenerator(samplewise_center=True,
                                 samplewise_std_normalization=True)                       
    train_gen = flow_from_dataframe(train_idg, train_df,
                                    path_col='path',
                                    y_col='xray14_vec',
                                    target_size=output_size,
                                    color_mode='grayscale',
                                    batch_size=batch_size)
    valid_gen = flow_from_dataframe(val_idg, valid_df,
                                    path_col='path',
                                    y_col='xray14_vec',
                                    target_size=output_size,
                                    color_mode='grayscale',
                                    batch_size=batch_size)
    return train_gen, valid_gen
    

def load_from_npy():
    """load dataset from numpy file"""
    train_val_df, test_df, xray14_labels = np.load('train_val_test_label.npy')
    return train_val_df, test_df, xray14_labels 


def split_patients_by_patient_ID(train_val_df, n_fold=5, seed=42):
    """random split train_val dataframe using patients id

    # Args
        train_val_df: dataframe, trianing and validation dataframe
        n_fold: int, split your all shuttfled patients id  in n folds, and first fold as validation patients id
        seed: int, set a random seed, get same reslut every time
    
    # Return
        train_df and val_df
    """
    np.random.seed(seed)
    patients_id = np.unique(train_val_df['Patient ID'])
    print('the length of patientd id is ', len(patients_id))
    val_len = int(len(patients_id) / n_fold)
    val_patients = np.random.choice(patients_id, val_len)
    trian_patients = [p_id for p_id in patients_id if p_id not in val_patients]
    trian_patients = np.array(trian_patients)
    print('the length of train patients is ', len(trian_patients))
    print('the length of val   patients is ', len(val_patients))
    train_df =  train_val_df.loc[train_val_df['Patient ID'].isin(trian_patients)].copy()
    val_df   =  train_val_df.loc[train_val_df['Patient ID'].isin(val_patients)].copy()

    return train_df, val_df


def get_class_weight(df, labels, multiply=1.0):
    """get 14 positive classes ratio, note that that the ratio is scaled to negitive class
    """
    def get_single_class_weight(pos_counts, total_counts):

        denominator = (total_counts - pos_counts * multiply) + pos_counts

        return (denominator - pos_counts) / denominator

    class_weights = [get_single_class_weight(df[c_label].sum(), len(df)) for c_label in labels]
    pos_w = np.array(class_weights, dtype='float32').flatten()
    pos_w = pos_w / (1.0 - pos_w)
    return pos_w


if __name__ == "__main__":
    # all_xray_df, xray8_labels, xray14_labels = load_data_entry('/home/share/data_repos/chest_xray')
    train_val_df, test_df, xray14_labels = load_from_text('/home/share/data_repos/chest_xray')
    np.save('train_val_test_label.npy', [train_val_df, test_df, xray14_labels])
