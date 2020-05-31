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
    def __init__(self, test_data, test_labels, classes, file_writer, figure_size=(12, 10)):
        super(ConfuseCallback, self).__init__()
        self.test_data = test_data
        self.test_labels = test_labels
        self.classes = classes
        self.file_writer = file_writer
        self.figure_size = figure_size

    def get_precision_recall_plot(self, con_mat):
        precisions = np.array([0] * len(self.classes)).astype('float32')
        recalls = np.array([0] * len(self.classes)).astype('float32')

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

        df = pd.DataFrame((self.classes, precisions, recalls)).T
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
        try:
            test_pred = self.model.predict(self.test_data)
            test_pred = tf.argmax(test_pred, axis=1)
            accuracy = np.sum(test_pred == self.test_labels) / self.test_labels.shape[0]
            con_mat = tf.math.confusion_matrix(labels=self.test_labels, predictions=test_pred).numpy()
            con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

            con_mat_df = pd.DataFrame(con_mat_norm,
                                      index=self.classes,
                                      columns=self.classes)

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
                     confuse_callback=True, label_info=None, test_generator_=None, figure_size=(12, 10),
                     modelsaver_callback=True, best_loss=float('inf'), save_root=None, best_epoch=0):
    postfix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root_ = "{}{}/".format(root, postfix)

    callbacks_ = []

    if tboard_callback:
        callbacks_.append(tf.keras.callbacks.TensorBoard(log_dir="{}fit".format(log_root_),
                                                         histogram_freq=tboard_histogram_freq,
                                                         update_freq=tboard_update_freq,
                                                         profile_batch=tboard_profile_batch)
                         )

    #TODO: generate label_info if not given
    if confuse_callback:
        file_writer = tf.summary.create_file_writer("{}/cm".format(log_root_, postfix))
        callbacks_.append(ConfuseCallback(test_generator_.data['test_data'],
                                          test_generator_.data['test_label'],
                                          list(label_info.keys()),
                                          file_writer, figure_size=figure_size)
                          )

    if modelsaver_callback:
        if not save_root:
            save_root = log_root_
        callbacks_.append(
            ModelSaverCallback(best_loss=best_loss, save_root=save_root, epoch=best_epoch)
        )

    return callbacks_, log_root_
