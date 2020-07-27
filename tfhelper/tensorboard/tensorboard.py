import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import io
from tensorboard import program
import datetime
import time
import os
import glob
import matplotlib.font_manager as fm


class ConfuseCallback(tf.keras.callbacks.Callback):
    """
    Generate Confusion Matrix and write an image to TensorBoard
    """
    def __init__(self, x_test, y_test, file_writer, dataset=None, class_names=None, figure_size=(12, 10), batch_size=32, model_out_idx=-1):
        """
        Args:
            x_test (None, np.ndarray): (n data, data dimension(Ex. 32x32x3 or 600x30 ..., etc). If None is given, dataset must be provided.
            y_test (None, np.ndarray): (n data, ). If None is given, dataset must be provided.
            file_writer (tf.summary.SummaryWriter): TensorBoard File Writer
            dataset (tf.keras.dataset.Dataset): If dataset is given, x_test and y_test is ignored. Default: None.
            class_names (list of str): Names of class. If None, default names are set to (Class01, Class02 ...). Default: None.
            figure_size (tuple): Figure size of confusion matrix. Default: (12, 10).
            batch_size (int): Batch size to predict x_test. If dataset is given, batch_size is ignored and batch size set in dataset is used.
            model_out_idx (int): If Model has multiple output, set this value for evaluation. -1: No output index
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
        self.model_out_idx = model_out_idx

        if self.label_names is None and self.y_test is not None:
            self.label_names = ["Class {:02d}".format(unique_label) for unique_label in np.unique(self.y_test)]

        # For Korean Label
        font_list = [font.name for font in fm.fontManager.ttflist]
        possible_fonts = [font_name for font_name in font_list if font_name.find("CJK") >=0]
        if len(possible_fonts) > 0:
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = possible_fonts[0]

    def get_precision_recall_plot(self, con_mat):
        """
        Generate Precision and Recall plot bar plot image
        Args:
            con_mat (np.ndarray): Confusion Matrix array

        Returns:
            tf.TensorArray: Precision and Recall Bar Plot Image
            np.ndarray: Precisions
            np.ndarray: Recalls
        """
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
                    if self.model_out_idx >= 0:
                        pred = pred[self.model_out_idx]

                    pred = np.argmax(pred, axis=1)
                    test_pred = np.concatenate([test_pred, pred])
            else:
                test_pred = self.model.predict(self.dataset)
                if self.model_out_idx >= 0:
                    test_pred = test_pred[self.model_out_idx]

                test_pred = np.argmax(test_pred, axis=1)

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
    """
    Saves Model at each end of the epoch when the best accuracy/loss is presented.
    """
    def __init__(self, best_metric=float('inf'), save_root="./", save_metric='val_loss', file_name="my_model", enable=True, epoch=0):
        """

        Args:
            best_metric (float): Set best score of previous training session if resuming.
            save_root (str): Model save path
            save_metric (str): One of 'val_loss', 'val_accuracy'
            enable (bool): Set previous epoch number if resuming
            epoch (int): Epoch number
        """
        super(ModelSaverCallback, self).__init__()
        self.best_metric = best_metric
        if self.best_metric == float('inf') and save_metric.find("accuracy") > 0:
            self.best_metric = -self.best_metric

        self.epoch = epoch
        self.save_root = save_root
        self.enable = enable
        self.save_metric = save_metric
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        try:
            epoch += self.epoch
            a = logs[self.save_metric]
            b = self.best_metric

            if self.save_metric.find("accuracy") > 0:
                a, b = b, a

            if a < b:
                p_file_list = glob.glob("{}/*.h5".format(self.save_root))
                p_file_list = sorted(p_file_list, key=lambda x: x[-10:])
                if self.save_metric.find("accuracy") < 0:
                    p_file_list = p_file_list[::-1]

                for i, file_path in enumerate(p_file_list):
                    if i+1 == len(p_file_list):
                        break
                    try:
                        os.remove(file_path)
                    except:
                        print("Error while deleting file : {}".format(file_path))

                file_name = '{}/{}_{:04d}_{}_{:03.2f}.h5'.format(self.save_root, self.file_name, epoch, self.save_metric, logs[self.save_metric])
                print("\nBest score! saving the model to {} ...".format(file_name))
                self.best_metric = logs[self.save_metric]

                if self.enable:
                    self.model.save(file_name)
        except Exception as e:
            print(e)

class SparsityCallback(tf.keras.callbacks.Callback):
    """
    Computes the sparsity on each layer of the given model and saves bar plot image to the TensorBoard.
    """
    def __init__(self, file_writer, sparsity_threshold=0.05, figure_size=(12, 20)):
        """

        Args:
            file_writer (tf.summary.SummaryWriter): TensorBoard File Writer
            sparsity_threshold (float): Sparsity Threshold of each layer.
                                        Ex) 0.05 -> Find the number of weights where -0.05 < values < 0.05 in a layer.
                                        Percentage of the number if set to the sparsity of the layer.
            figure_size (tuple): Figure size to generate plot image
        """
        super(SparsityCallback, self).__init__()

        self.file_writer = file_writer
        self.sparsity_threshold = sparsity_threshold
        self.figure_size = list(figure_size)

    def get_sparsity_plot(self, sparse_levels, sparse_layer_names):
        """
        Generate sparsity plot image

        Args:
            sparse_levels (np.ndarray): Sparse levels for the layer
            sparse_layer_names (list of str, np.ndarray): Names of layer along with sparse_levels list

        Returns:

        """
        width = 0.8

        n_data = sparse_levels.shape[0]
        self.figure_size[1] = 0.25 * n_data

        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.barh(sparse_layer_names, sparse_levels, width)

        for i, v in enumerate(sparse_levels):
            ax.text(v + 0.005, i - .15, f"{v * 100:.2f}%", color='k', fontweight='bold')

        ax.set_title(f"Sparsity Threshold: {self.sparsity_threshold}, Mean Sparsity: {sparse_levels.mean()*100:.2f}%")
        ax.set_xlim(0.0, 1.0)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        return image

    def on_epoch_end(self, epoch, logs=None):
        sparsities = self.compute_sparsity()
        layer_names = [layer.name for layer in self.model.layers]

        sparse_levels = []
        sparse_layer_names = []

        for i, (sparsity, layer_name) in enumerate(zip(sparsities, layer_names)):
            if np.isnan(sparsity) or sparsity == 0.0:
                continue

            sparse_levels = np.concatenate([sparse_levels, [sparsity]])
            sparse_layer_names = np.concatenate([sparse_layer_names, [f"{i:03d}: {layer_name}"]])

        try:
            sparsity_image = self.get_sparsity_plot(sparse_levels, sparse_layer_names)

            with self.file_writer.as_default():
                tf.summary.image("Sparsity Levels of Each Layer", sparsity_image, step=epoch)
        except Exception as e:
            print(e)

    def compute_sparsity(self):
        sparsities = np.zeros(len(self.model.layers))

        for i in range(sparsities.shape[0]):
            if len(self.model.layers[i].weights) < 1:
                sparsities[i] = np.nan
                continue

            sparse_index = np.argwhere(
                np.logical_and(self.model.layers[i].weights[0].numpy().flatten() < self.sparsity_threshold,
                               self.model.layers[i].weights[0].numpy().flatten() > -self.sparsity_threshold))

            sparsities[i] = sparse_index.shape[0] / np.prod(self.model.layers[i].weights[0].shape)

        return sparsities


def run_tensorboard(path, host='0.0.0.0', port=6006):
    """
    Run TensorBoard in python script.
    Args:
        path (str): TensorBoard log dir
        host(str): Host address for TensorBoard.
                    127.0.0.1 -> localhost.
                    0.0.0.0 -> Allow remote connection.
        port (int): Port number for TensorBoard

    Returns:
        None
    """
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', path, '--host', host, '--port', f"{port:}"])
    url = tb.launch()

    print("Running tensorboard on {}".format(url))

    return url


def wait_ctrl_c(pre_msg="Press Ctrl+c to quit Tensorboard", post_msg="\nExit."):
    """
    Wait until ctrl+c is pressed. This function is to prevent quitting python process when the training is completed when TensorBoard is running.
    Args:
        pre_msg: Message prior to wait ctrl+c
        post_msg: Message post to ctrl+c pressed

    Returns:
        None
    """
    print(pre_msg)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print(post_msg)


def get_tf_callbacks(root,
                     tboard_callback=True, tboard_update_freq='epoch', tboard_histogram_freq=1, tboard_profile_batch=0,
                     confuse_callback=True, label_info=None, x_test=None, y_test=None, test_generator_=None, test_dataset=None, figure_size=(12, 10), model_out_idx=-1,
                     modelsaver_callback=False, best_loss=float('inf'), save_root=None, best_epoch=0, save_metric='val_loss', save_file_name="my_model",
                     earlystop_callback=True, earlystop_monitor='val_loss', earlystop_patience=0, earlystop_restore_weights=True,
                     sparsity_callback=False, sparsity_threshold=0.05):
    """
    Getting TensorFlow callbacks function for convenience purpose.
    Args:
        root (str): Root directory for TensorBoard
        tboard_callback (bool): Whether using TensorBoard or not. Default: True
        tboard_update_freq (str): TensorBoard update frequency. ('epoch', 'batch'). Default: 'epoch'
        tboard_histogram_freq (int): TensorBoard histogram update frequency. Default: 1
        tboard_profile_batch (int): TensorBoard profile timing. If 0 is given, profiling is not used.
                                Ex) If 10 is given, profiling is executed at batch of 10. Default: 0
        confuse_callback (bool): Whether using confusion matrix for TensorBoard callback or not.
                          At least one of the following three ((x_test, y_test), test_generator_, test_dataset) must be set.
                          Otherwise, Confusion Matrix callback will be ignored.
                          Default: True.
        label_info (list of str): Names of class. If None, default names are set to (Class01, Class02 ...). Default: None.
        x_test (np.ndarray, None): (n data, data dimension(Ex. 32x32x3 or 600x30 ..., etc). If None is given, dataset must be provided.
        y_test (np.ndarray, None): (n data, ). If None is given, dataset must be provided.
        test_generator_ (tfhelper.dataset.HDF5Generator, None): Default: None. For HDF5Generator test set purpose.
        test_dataset (tf.dataset.Dataset, None): Default: None.
        figure_size (tuple): Figure Size of Confusion Matrix.
        model_out_idx (int): If Model has multiple output, set this value for evaluation. -1: No output index
        modelsaver_callback (bool): Whether using ModelSaver callback or not. Saving the model file when the lowest validation loss is given per each epochs.
                                        Default: False.
        best_loss (float): Set best score of previous training session if resuming.
        save_root (str): Model save path
        best_epoch (int): Previous Best epoch number if resuming
        save_metric (str): One of 'val_loss', 'val_accuracy'
        save_file_name (str): Model Save Target File Name
        earlystop_callback (bool): Early Stop callback
        earlystop_monitor (str): Earlys top_monitor metric 'val_loss', 'val_accuracy'
        earlystop_patience (int): Early stop patience
        earlystop_restore_weights (bool): Restore weights on early stop.
        sparsity_callback (bool): Sparsity callback.
        sparsity_threshold (float): Sparsity Threshold of each layer.
                            Ex) 0.05 -> Find the number of weights where -0.05 < values < 0.05 in a layer.
                            Percentage of the number if set to the sparsity of the layer.

    Returns:
        list of tf.keras.callbacks.Callback: Callback List
        str: Tensor Board Log Root Directory

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
            callbacks_.append(ConfuseCallback(None, y_test, file_writer, dataset=test_dataset, class_names=label_info,
                                              figure_size=figure_size, model_out_idx=model_out_idx)
                              )

    if modelsaver_callback:
        if not save_root:
            save_root = log_root_
        callbacks_.append(
            ModelSaverCallback(best_metric=best_loss, save_root=save_root, epoch=best_epoch, save_metric=save_metric, file_name=save_file_name)
        )

    if earlystop_callback:
        callbacks_.append(tf.keras.callbacks.EarlyStopping(monitor=earlystop_monitor, patience=earlystop_patience,
                                                           restore_best_weights=earlystop_restore_weights))

    if sparsity_callback:
        file_writer = tf.summary.create_file_writer("{}/sparsity".format(log_root_, postfix))

        callbacks_.append(
            SparsityCallback(file_writer, sparsity_threshold=sparsity_threshold)
        )

    return callbacks_, log_root_
