#encoding: utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import range
import warnings
from keras import utils as keras_utils
import threading
from keras import backend as K
from .image import load_img, img_to_array
import random

class Iterator(keras_utils.Sequence):
    """Base class for image data iterators.

    every derived class must implement `_get_batches_of_transformed_samples` method

    # Args
        n: int, num of samples.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, if shuttfe sample index in epoch end.
            if set it True, instance is ordered iterator
        seed: Random seeding for data shuffling.
    """
    def __init__(self, n, batch_size, seed, shuffle=False):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()   # 生成器

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        # 设置随机数种子,随着batch index改变
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen:
        return self

    def __next__(self, *args, **kwargs):

        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class ImageGeneratorFromPath(Iterator):
    """Iterator yielding data from a Numpy array.
    can use for predict_gennerator
    # Arguments
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
                 suggest when training, set it True, testing set is False
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
    """

    def __init__(self, image_transformer,
                 path_df, label_df,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 target_size=(256, 256),
                 data_format=None,
                 color_mode='grayscale'):
        self.paths = list(path_df.values)
        self.labels = np.stack(label_df.values)
        assert len(self.paths) == len(self.labels)
        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        self.tansformer = image_transformer
        self.target_size = target_size
        self.color_mode = color_mode
        super(ImageGeneratorFromPath, self).__init__(len(self.paths),
                                                          batch_size,
                                                          shuffle,
                                                          seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.color_mode == 'rgb':
            channel = 3
        else:
            channel = 1

        image_shape = list(self.target_size) + [channel]
        batch_x = np.zeros(tuple([len(index_array)] + image_shape),
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            img = load_img(self.paths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation='bilinear')
            x = img_to_array(img, data_format=self.data_format)
            if hasattr(img, 'close'):
                img.close()
            params = self.tansformer.get_random_transform(image_shape)
            x = self.tansformer.apply_transform(x.astype(K.floatx()), params)
            x = self.tansformer.standardize(x)
            batch_x[i] = x
        batch_y = self.labels[index_array]
        self.cur_index = index_array
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def get_label(self):
        
        return self.labels

def random_image_generator(transformer, df, target_size, xray_labels, batch_size, color_mode):
    """batch images generator

    # Args
        transformer: class, include kinds of image transforming and augmenting methods.
        df: dataframe, must have 'path' (image path) and 'xray14_vec' (image class label) column
        target_size: tuple or list, original image will be resized to target size as model batch input image
        color_mode: str, rgb channel or grayscale image, if 'rgb', image will be stacked 3 channels

    # Return
        batch_x: yileded batch image, (batch_size, height, width, channel)
        batch_y: yileded batch label, (batch_size, 14), 14 is number of classes
    """
    if color_mode == 'rgb':
        channel = 3
    else:
        channel = 1

    image_shape = list(target_size) + [channel]
    batch_x = np.zeros(tuple([batch_size] + image_shape), dtype=K.floatx())
    batch_y = np.zeros(tuple([batch_size] + [len(xray_labels)]), dtype=K.floatx())
    while 1:
        for i in range(batch_size):
            # disease = random.choice(xray_labels)
            sample = df.sample(1)
            path = sample['path'].values[0]
            img = load_img(path,
                           color_mode=color_mode,
                           target_size=target_size,
                           interpolation='bilinear')
            x = img_to_array(img, data_format=K.image_data_format())
            if hasattr(img, 'close'):
                img.close()
            params = transformer.get_random_transform(image_shape)
            x = transformer.apply_transform(x.astype(K.floatx()), params)
            x = transformer.standardize(x)
            batch_x[i, :, :, :] = x
            label = np.array(sample['xray14_vec'].values[0], dtype=np.float32)
            batch_y[i, :] = label

        yield batch_x, batch_y


def image_generator_by_diseases(transformer, df, target_size, xray_labels, batch_size,
                                color_mode):
    """batch images generator

    # Args
        transformer: class, include kinds of image transforming and augmenting methods.
        df: dataframe, must have 'path' (image path) and 'xray14_vec' (image class label) column
        target_size: tuple or list, original image will be resized to target size as model batch input image
        color_mode: str, rgb channel or grayscale image, if 'rgb', image will be stacked 3 channels

    # Return
        batch_x: yileded batch image, (batch_size, height, width, channel)
        batch_y: yileded batch label, (batch_size, 14), 14 is number of classes
    """
    if color_mode == 'rgb':
        channel = 3
    else:
        channel = 1

    image_shape = list(target_size) + [channel]
    batch_x = np.zeros(tuple([batch_size] + image_shape), dtype=K.floatx())
    batch_y = np.zeros(tuple([batch_size] + [len(xray_labels)]), dtype=K.floatx())
    while 1:
        for i in range(batch_size):
            # random select a class
            disease = random.choice(xray_labels)
            sample = df[df[disease] > 0].sample(1)
            path = sample['path'].values[0]
            img = load_img(path,
                           color_mode=color_mode,
                           target_size=target_size,
                           interpolation='bilinear'
                           )
            x = img_to_array(img, data_format=K.image_data_format())
            if hasattr(img, 'close'):
                img.close()
            params = transformer.get_random_transform(image_shape)
            x = transformer.apply_transform(x.astype(K.floatx()), params)
            x = transformer.standardize(x)
            batch_x[i] = x
            label = sample['xray14_vec'].values[0]
            batch_y[i] = label
        yield batch_x, batch_y