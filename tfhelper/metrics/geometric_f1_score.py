import tensorflow as tf
from tfhelper.utils import tfprint
import numpy as np


class GeometricF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="geometric_f1score", n_classes=10, debug=False, skip_nan=False, **kwargs):
        super(GeometricF1Score, self).__init__(name=name, **kwargs)
        self.f1_scores = self.add_weight(name="f1score", initializer='zeros')
        self.con_mat = tf.Variable(initial_value=np.zeros((n_classes, n_classes)), dtype=tf.float32, name="geometric_f1score_con_mat")
        self.n_classes = n_classes
        self.debug = debug
        self.skip_nan = skip_nan

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int32)
        y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=y_true.dtype)

        con_mat = tf.cast(tf.math.confusion_matrix(labels=y_true,
                                                   predictions=y_pred,
                                                   num_classes=self.n_classes),
                          dtype=tf.float32)
        self.con_mat.assign_add(con_mat)

        tp = tf.linalg.diag_part(self.con_mat)
        fp = tf.reduce_sum(self.con_mat, axis=0) - tp
        fn = tf.reduce_sum(self.con_mat, axis=1) - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        precision = tf.where(tf.math.is_nan(precision), tf.zeros_like(precision), precision)
        recall = tf.where(tf.math.is_nan(recall), tf.zeros_like(recall), recall)

        if self.skip_nan:
            idx0 = tf.reduce_sum(con_mat, axis=0) > 0
            idx1 = tf.reduce_sum(con_mat, axis=1) > 0
            idx = tf.reshape(tf.where(tf.logical_or(idx0, idx1)), [-1])

            precision = tf.gather(precision, idx, axis=0)
            recall = tf.gather(recall, idx, axis=0)

        f1score = tf.sqrt(precision * recall)

        if self.debug:
            tf.print("")
            for i in range(self.n_classes):
                tfprint(self.con_mat[i], len=self.n_classes, name=f"con_mat[{i:02d}]", format="{:>7,}, ")

            tfprint(y_true, name="y_true", format="{:02d}, ")
            tfprint(y_pred, name="y_pred", format="{:02d}, ")
            tfprint(precision, name="precision", format="{:.05f}, ")
            tfprint(recall, name="recall", format="{:.05f}, ")
            tfprint(f1score, name="f1scores", format="{:.05f}, ")

        f1score = tf.sqrt(tf.math.reduce_prod(f1score))
        self.f1_scores.assign(f1score)

    def reset_states(self):
        self.con_mat.assign(np.zeros((self.n_classes, self.n_classes)))
        self.f1_scores.assign(0.0)

    def result(self):
        return self.f1_scores