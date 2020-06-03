import tensorflow as tf
import numpy as np
import cv2


def get_cam_image(model_, x, img_size=(28, 28), layer_idx=None):
    if layer_idx is None:
        for layer_idx in range(len(model_.layers) - 1, -1, -1):
            if type(model_.layers[layer_idx]) == tf.keras.layers.Conv2D:
                break

    cam_model_ = tf.keras.models.Model(model_.inputs, [model_.layers[layer_idx].output, model_.output])
    conv_out, model_out = cam_model_(x)

    cam_images_ = np.zeros((x.shape[0], img_size[0], img_size[1]))

    for i, outs in enumerate(zip(conv_out, model_out)):
        c_out, m_out = outs
        predict_idx = np.argmax(m_out)
        chosen_weight = model_.layers[-1].weights[0][:, predict_idx]

        cam_img_ = np.zeros(c_out.shape[0:2])

        for j in range(c_out.shape[2]):
            cam_img_ += c_out[:, :, j] * chosen_weight[j]

        cam_images_[i] = cv2.resize(cam_img_.numpy(), img_size)

    return cam_images_