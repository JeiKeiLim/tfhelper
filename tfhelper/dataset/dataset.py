import h5py
import numpy as np
import tensorflow as tf


class HDF5Generator:
    """
    TensorFlow Generator with h5py file format
    """

    def __init__(self, path, data_field_name, label_field_name, verbose=1):
        """
        Args:
            path (str): HDF5 dataset path
            data_field_name (str): data field name inside hdf5 file. ex) "train_data" or "test_data"
            label_field_name (str): label field name inside hdf5 file. ex) "train_label" or "test_label"
            verbose (int):
        """
        self.path = path
        self.data = h5py.File(self.path, 'r')
        self.data_field_name = data_field_name
        self.label_field_name = label_field_name

        self.n_data = self.data[self.data_field_name].shape[0]

        labels = self.data[self.label_field_name][()]
        unique_labels = np.unique(labels)
        self.class_histogram = np.array([labels[labels == i].shape[0] for i in unique_labels])
        self.n_class = unique_labels.shape[0]

        c_weight = (1 / self.class_histogram) * labels.shape[0] / self.n_class
        self.class_weight = {i: weight for i, weight in enumerate(c_weight)}

        if verbose > 0:
            print("=+*"*20)
            print("Dataset {} with {} data".format(path, self.n_data))
            print("=+*"*20)

    def __call__(self):
        with h5py.File(self.path, 'r') as f:
            for d, l in zip(f[self.data_field_name], f[self.label_field_name]):
                yield (d, l)

    def get_dataset(self, input_shape=(300, 30), batch_size=32, shuffle=False, n_shuffle=10000):
        """
        Get tf.data.Dataset type from the generator. Currently, data and label types are tf.float32 and tf.int8
        Args:
            input_shape (tuple): Input shape
            batch_size (int): Batch size
            shuffle (bool): Shuffle
            n_shuffle (int): buffer size for shuffle

        Returns:
            tf.data.Dataset: TensorFlow Dataset
        """

        d_set = tf.data.Dataset.from_generator(
            self,
            (tf.float32, tf.int8),
            (tf.TensorShape(input_shape), tf.TensorShape([]))
        )
        d_set = d_set.shuffle(n_shuffle, reshuffle_each_iteration=True) if shuffle else d_set

        return d_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()

    def nan_check(self):
        """
        Check if nan exists in self.data
        Returns:
            bool: True or False
        """

        is_nan = False
        data_field = self.data[self.data_field_name]
        for i in range(data_field.shape[0]):
            if np.sum(np.isnan(data_field[i].flatten())) > 0:
                print("NaN Exist! on {}".format(i))
                is_nan = True
        return is_nan
