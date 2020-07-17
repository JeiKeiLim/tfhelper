import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def predict_tflite_interpreter(interpreter, x_, predict_class=True):
    """
    Predict x_ with tflite interpreter

    Args:
        interpreter (tf.lite.Interpreter): TF Lite Interpreter
        x_ (np.ndarray): test data
        predict_class (bool): True: return argmax(result).
                              False: return as is
    Returns:
        np.array: Predicted Label or Values of top layer.
    """
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, np.expand_dims(x_, axis=0).astype(np.float32))
    interpreter.invoke()
    prediction_ = interpreter.get_tensor(output_index)

    return np.argmax(prediction_), prediction_ if predict_class else prediction_


def evaluate_tflite_interpreter(interpreter, x_test_, y_test_):
    """
    Evaluate tflite interpreter

    Args:
        interpreter (tf.lite.Interpreter): TF Lite Interpreter
        x_test_ (np.ndarray): test data
        y_test_ (np.ndarray): test label

    Returns:
        float: Accuracy
        np.ndarray: Prediction Results
    """
    prediction_result = []
    for x in x_test_:
        predict_label, _ = predict_tflite_interpreter(interpreter, x)
        prediction_result.append(predict_label)

    prediction_result = np.array(prediction_result)
    accuracy = np.sum(prediction_result == y_test_.flatten()) / y_test_.shape[0]

    return accuracy, prediction_result


def load_pruned_model(file_path, strip_model=True):
    """
    Load pruned TensorFlow Keras model
    :param file_path:
    :type file_path: str
    :param strip_model: True if the saved model is stripped.
    :type strip_model: bool
    :return:
    """
    model_ = tf.keras.models.load_model(file_path, custom_objects={
        'PruneLowMagnitude': tfmot.sparsity.keras.pruning_wrapper.PruneLowMagnitude})
    model_ = tfmot.sparsity.keras.strip_pruning(model_) if strip_model else model_

    return model_
