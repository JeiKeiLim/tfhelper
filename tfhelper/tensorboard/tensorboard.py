import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import io
from tensorboard import program
import datetime
import time


class ConfuseCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, file_writer, dataset=None, class_names=None, figure_size=(12, 10), batch_size=32):
        """

        :param x_test: (array) - (n data, data dimension(Ex. 32x32x3 or 600x30 ..., etc). If None is given, dataset must be provided.
        :param y_test: (array) - (n data, ). If None is given, dataset must be provided.
        :param file_writer: TensorBoard File Writer
        :param dataset: (TensorFlow Dataset). If dataset is given, x_test and y_test is ignored. Default: None.
        :param class_names: (array(str)) - Names of class. If None, default names are set to (Class01, Class02 ...). Default: None.
        :param figure_size: (tuple) - Figure size of confusion matrix. Default: (12, 10).
        :param batch_size: (int) - Batch size to predict x_test. If dataset is given, batch_size is ignored and batch size set in dataset is used.
        """
        super(ConfuseCallback, self).__init__()
        self.dataset = dataset
        self.x_test = x_test
        self.y_test = y_test

        if self.y_test is None and self.dataset is not None:
            self.y_test = []
            for i, xy in self.dataset.enumerate():
                y = xy[1].numpy()
                self.y_test = np.concatenate([self.y_test, y])
            self.y_test = self.y_test.astype(np.int32)

        self.y_test = self.y_test if len(self.y_test.shape) == 1 else np.argmax(self.y_test, axis=1)

        self.file_writer = file_writer
        self.figure_size = figure_size
        self.label_names = class_names
        self.batch_size = batch_size

        if self.label_names is None and self.y_test is not None:
            self.label_names = ["Class {:02d}".format(unique_label) for unique_label in np.unique(self.y_test)]

    def get_precision_recall_plot(self, con_mat):
        precisions = np.array([0] * len(self.label_names)).astype('float32')
        recalls = np.array([0] * len(self.label_names)).astype('float32')

        for i in range(con_mat.shape[0]):
            tp = con_mat[i, i]
            fn = (con_mat[i, :].sum() - tp)

            fp = (con_mat[:, i].sum() - tp)
            tn = (con_mat.diagonal().sum() - tp)

            # tpr = tp / np.sum(self.test_labels[()] == i)
            # fnr = fn / np.sum(self.test_labels[()] == i)
            # fpr = fp / np.sum(self.test_labels[()] != i)
            # tnr = tn / np.sum(self.test_labels[()] != i)

            precisions[i] = max(0, tp / (tp + fp))
            recalls[i] = max(0, tp / (tp + fn))

        df = pd.DataFrame((self.label_names, precisions, recalls)).T
        df.columns = ["Class", "Precision", "Recall"]
        df = pd.melt(df, id_vars="Class", var_name="Type", value_name="Value")

        figure = plt.figure(figsize=self.figure_size)
        sns.barplot(y='Class', x='Value', hue='Type', data=df)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        return image, precisions, recalls

    def on_epoch_end(self, epoch, logs=None):
        if self.dataset is None and (self.x_test is None or self.y_test is None):
            return

        try:
            if self.dataset is None:
                test_pred = []
                for b in range(0, self.x_test.shape[0], self.batch_size):
                    x_feed = self.x_test[b:b+self.batch_size]
                    pred = self.model.predict(x_feed)
                    pred = np.argmax(pred, axis=1)
                    test_pred = np.concatenate([test_pred, pred])
            else:
                test_pred = self.model.predict(self.dataset)

            accuracy = np.sum(test_pred == self.y_test) / self.y_test.shape[0]

            con_mat = tf.math.confusion_matrix(labels=self.y_test, predictions=test_pred).numpy()
            con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

            con_mat_df = pd.DataFrame(con_mat_norm,
                                      index=self.label_names,
                                      columns=self.label_names)

            precision_recall_image, precisions, recalls = self.get_precision_recall_plot(con_mat)

            figure = plt.figure(figsize=self.figure_size)
            sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title("Accuracy : {:.2f}%, Precision : {:.2f}%, Recall : {:.2f}%".format(accuracy*100, precisions.mean()*100, recalls.mean()*100))

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')

            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)

            image = tf.expand_dims(image, 0)

            # Log the confusion matrix as an image summary.
            with self.file_writer.as_default():
                tf.summary.image("Confusion Matrix", image, step=epoch)
                tf.summary.image("Precision and Recall", precision_recall_image, step=epoch)
        except Exception as e:
            print(e)


class ModelSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, best_loss=float('inf'), save_root="./", enable=True, epoch=0):
        super(ModelSaverCallback, self).__init__()
        self.best_loss = best_loss
        self.epoch = epoch
        self.save_root = save_root
        self.enable = enable

    def on_epoch_end(self, epoch, logs=None):
        try:
            epoch += self.epoch
            if logs['val_loss'] < self.best_loss:
                # TODO Delete previous saved model if exists
                file_name = '{}my_model_weight_{:04d}_{:03.2f}.h5'.format(self.save_root, epoch, logs['val_loss'])
                print("\nBest loss! saving the model to {} ...".format(file_name))
                self.best_loss = logs['val_loss']

                if self.enable:
                    self.model.save(file_name)
        except Exception as e:
            print(e)


def run_tensorboard(path, host='0.0.0.0', port=6006):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', path, '--host', host, '--port', f"{port:}"])
    url = tb.launch()

    print("Running tensorboard on {}".format(url))

    return url


def wait_ctrl_c(pre_msg="Press Ctrl+c to quit Tensorboard", post_msg="\nExit."):
    print(pre_msg)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print(post_msg)


def get_tf_callbacks(root,
                     tboard_callback=True, tboard_update_freq='epoch', tboard_histogram_freq=1, tboard_profile_batch=0,
                     confuse_callback=True, label_info=None, x_test=None, y_test=None, test_generator_=None, test_dataset=None, figure_size=(12, 10),
                     modelsaver_callback=False, best_loss=float('inf'), save_root=None, best_epoch=0,
                     earlystop_callback=True, earlystop_monitor='val_loss', earlystop_patience=0, earlystop_restore_weights=True):
    """
    Get TensorFlow callbacks function for convinience
    :param root: Root directory for TensorBoard
    :param tboard_callback: (bool) - Whether using TensorBoard or not. Default: True
    :param tboard_update_freq: (str) - TensorBoard update frequency. ('epoch', 'batch'). Default: 'epoch'
    :param tboard_histogram_freq: (int) - TensorBoard histogram update frequency. Default: 1
    :param tboard_profile_batch: (int) - TensorBoard profile timing. If 0 is given, profiling is not used.
                                        Ex) If 10 is given, profiling is executed at batch of 10. Default: 0
    :param confuse_callback: (bool) - Whether using confusion matrix for TensorBoard callback or not.
                                      At least one of the following three ((x_test, y_test), test_generator_, test_dataset) must be set.
                                      Otherwise, Confusion Matrix callback will be ignored.
                                      Default: True.
    :param label_info: (None or array(str))-  Default: None
    :param x_test: (None or array) - Default: None
    :param y_test: (None or array) = Default: None
    :param test_generator_: (None or HDF5Generator) - Default: None. For HDF5Generator test set purpose.
    :param test_dataset: (None or TensorFlow Dataset) - Default: None.
    :param figure_size: (tuple) - Figure Size of Confusion Matrix.
    :param modelsaver_callback: (bool) - Whether using ModelSaver callback or not. Saving the model file when the lowest validation loss is given per each epochs.
                                        Default: False.
    :param best_loss:
    :param save_root:
    :param best_epoch:
    :param earlystop_callback:
    :param earlystop_monitor:
    :param earlystop_patience:
    :param earlystop_restore_weights:
    :return:
    """
    postfix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root_ = "{}{}/".format(root, postfix)

    callbacks_ = []

    if tboard_callback:
        callbacks_.append(tf.keras.callbacks.TensorBoard(log_dir="{}fit".format(log_root_),
                                                         histogram_freq=tboard_histogram_freq,
                                                         update_freq=tboard_update_freq,
                                                         profile_batch=tboard_profile_batch)
                         )

    if confuse_callback:
        file_writer = tf.summary.create_file_writer("{}/cm".format(log_root_, postfix))

        x_test = test_generator_.data['test_data'] if test_generator_ is not None else x_test
        y_test = test_generator_.data['test_label'] if test_generator_ is not None else y_test

        if x_test is not None and y_test is not None:
            callbacks_.append(ConfuseCallback(x_test, y_test, file_writer, class_names=label_info,
                                              figure_size=figure_size)
                              )
        elif test_dataset is not None:
            callbacks_.append(ConfuseCallback(None, None, file_writer, dataset=test_dataset, class_names=label_info,
                                              figure_size=figure_size)
                              )

    if modelsaver_callback:
        if not save_root:
            save_root = log_root_
        callbacks_.append(
            ModelSaverCallback(best_loss=best_loss, save_root=save_root, epoch=best_epoch)
        )

    if earlystop_callback:
        callbacks_.append(tf.keras.callbacks.EarlyStopping(monitor=earlystop_monitor, patience=earlystop_patience,
                                                           restore_best_weights=earlystop_restore_weights))

    return callbacks_, log_root_
