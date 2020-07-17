# tfhelper
Your friendly Tensorflow 2.x neighbor.

Documentation: https://jeikeilim.github.io/tfhelper/


## Environments
- Python 3.7

# Getting started
## pip install
```
pip install tfhelper
```


# Structure
## tfhelper.dataset
- HDF5Generator
```python
from tfhelper.dataset import HDF5Generator

train_generator = HDF5Generator("/dataset/path.hdf5", "training_data", "test_data")
train_dataset = train_generator.get_dataset(input_shape=(300, 300), batch_size=16, shuffle=True, n_shuffle=1000)

...
model = Define some model

model.fit(train_dataset)
```

## tfhelper.gpu
- allow_gpu_memory_growth
## tfhelper.tensorboard
- ConfuseCallback
- ModelSaverCallback
- SparsityCallback
- run_tensorboard
- wait_ctrl_c
- get_tf_callbacks
## tfhelper.tflite
- keras_model_to_tflite
- parse_config
- predict_tflite_interpreter
- evaluate_tflite_interpreter
- load_pruned_model
## tfhelper.transfler_learning
- get_transfer_learning_model
## tfhelper.visualization
- get_cam_image



