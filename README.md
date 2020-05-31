# tfhelper
Your friendly Tensorflow 2.x neighbor.

# Usage
## dataset
### HDF5Generator
```python
from tfhelper.dataset import HDF5Generator

train_generator = HDF5Generator("/dataset/path.hdf5", "training_data", "test_data")
train_dataset = train_generator.get_dataset(input_shape=(300, 300), batch_size=16, shuffle=True, n_shuffle=1000)

...
model = Define some model

model.fit(train_dataset)
```

## tensorboard
### ConfuseCallback
### ModelSaverCallback
### run_tensorboard
### wait_ctrl_c
### get_tf_callbacks

## gpu
### allow_gpu_memory_growth

## transfer_learning
### get_transfer_learning_model

# Getting started
## pip install
```
pip install tfhelper
```

## Environments
- Python 3.7

