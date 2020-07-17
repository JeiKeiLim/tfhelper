import tensorflow as tf


def keras_model_to_tflite(model, config):
    """
    Convert Keras model to tflite model

    Args:
        model (tf.keras.models.Model): TensorFlow Model
        config (dict): configuration info Ex)
                        {
                          "quantization": false,
                          "quantization_type": "int8",  # ["int8", "float16", "float32"]
                          "tf_ops": false,
                          "exp_converter": false,
                          "out_path": "/writing/tflite/model/path"
                        }

    Returns:
        The converted tflite model data in serialized format.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    config = parse_config(config)

    optimizations = []
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    supported_types = []

    if config['quantization']:
        optimizations += [tf.lite.Optimize.DEFAULT]
        supported_types += [config['quantization_type']]

        if config['quantization_type'] == tf.uint8:
            supported_ops += tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

    if config['tf_ops']:
        supported_ops += [tf.lite.OpsSet.SELECT_TF_OPS]

    converter.optimizations = optimizations
    converter.target_spec.supported_ops = supported_ops
    converter.target_spec.supported_types = supported_types

    converter.experimental_new_converter = config['exp_converter']

    tflite_model = converter.convert()

    with open(config['out_path'], 'wb') as f:
        f.write(tflite_model)

    print('Saved TFLite model to:', config['out_path'])

    return tflite_model


def parse_config(config):
    """
    Parse config dict in keras_model_to_tflite
    Converts data type written as str to tf dtypes
    'float32' -> tf.float32
    'float16' -> tf.float16
    'int8' -> tf.int8

    Args:
        config (dict): tflite config dict from keras_model_to_tflite
    Returns:
        dict: Converted config
    """
    qtype = config['quantization_type']
    config['quantization_type'] = tf.float16 if qtype == "float16" else tf.int8 if qtype == "int8" else tf.float32

    return config
