import numpy as np
import tensorflow as tf

class DatasetTransposeWrapper(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def random_batch(self, batch_size):
        obs, a = self.dataset.random_batch(batch_size)
        obs = np.transpose(obs, [0, 1, 3, 2, 4])
        return obs, a

    def transform(x):
        # x [None, 64, 64, 1]
        x = tf.transpose(x, [0, 2, 1, 3])
        return x

    def data_transform(x):
        # x [None, 64, 64, 1]
        x = np.transpose(x, [0, 2, 1, 3])
        return x

class DatasetSwapWrapper(object):
    # swap left and right
    def __init__(self, dataset):
        self.dataset = dataset

    def random_batch(self, batch_size):
        obs, a = self.dataset.random_batch(batch_size)
        obs = obs[:, :, :, ::-1, :] # left <-> right 
        return obs, a

    def transform(x):
        x = x[:, :, ::-1, :]
        return x

    def data_transform(x):
        x = x[:, :, ::-1, :]
        return x

class DatasetHorizontalConcatWrapper(object):
     # split vertically and concat horizontally in inverse order 
    def __init__(self, dataset):
        self.dataset = dataset

    def random_batch(self, batch_size):
        obs, a = self.dataset.random_batch(batch_size)
        sub_obs = np.split(obs, 2, axis=3)
        obs = np.concatenate(sub_obs[::-1], axis=3)
        return obs, a

    def transform(x):
        xs = tf.split(x, 2, axis=2)
        x = tf.concat(xs[::-1], axis=2)
        return x


    def data_transform(x):
        xs = np.split(x, 2, axis=2)
        x = np.concatenate(xs[::-1], axis=2)
        return x

class DatasetVerticalConcatWrapper(object):
     # split horizontally and concat vertically in inverse order 
    def __init__(self, dataset):
        self.dataset = dataset

    def random_batch(self, batch_size):
        obs, a = self.dataset.random_batch(batch_size)
        sub_obs = np.split(obs, 2, axis=2)
        obs = np.concatenate(sub_obs[::-1], axis=2)
        return obs, a

    def transform(x):
        xs = tf.split(x, 2, axis=1)
        x = tf.concat(xs[::-1], axis=1)
        return x

    def data_transform(x):
        xs = np.split(x, 2, axis=1)
        x = np.concatenate(xs[::-1], axis=1)
        return x


class DatasetColorWrapper(object):
     # split horizontally and concat vertically in inverse order 
    def __init__(self, dataset):
        self.dataset = dataset

    def random_batch(self, batch_size):
        obs, a = self.dataset.random_batch(batch_size)
        return 1.0-obs, a

    def transform(x):
        return 1.0-x

    def data_transform(x):
        return 1.0-x
