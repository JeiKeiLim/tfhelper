import tensorflow as tf


def allow_gpu_memory_growth(gpu_idx=0):
    """
    Allowing GPU memory to grow as Tensorflow requires.
    :param gpu_idx: Index number of the target GPU
    :type gpu_idx: int
    :return: None
    :rtype: None
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[gpu_idx], True)
        print("Success allowing gpu memory growth on {:02d} gpu device".format(gpu_idx))
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("Failed to allow gpu memory growth on {:02d} gpu device".format(gpu_idx))

