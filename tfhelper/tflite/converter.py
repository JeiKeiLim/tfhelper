import tensorflow as tf


def keras_model_to_tflite(model, config):
    """

    :param model:
    :param config: (dict) configuration info Ex)
    {
      "quantization": false,
      "quantization_type": "int8",  # ["int8", "float16", "float32"]
      "tf_ops": false,
      "exp_converter": false,
      "out_path": "/writing/tflite/model/path"
    }
    :return:
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

    qtype = config['quantization_type']
    config['quantization_type'] = tf.float16 if qtype == "float16" else tf.int8 if qtype == "int8" else tf.float32

    return config
