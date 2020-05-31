import tensorflow as tf


def get_transfer_learning_model(target_model=tf.keras.applications.VGG19,
                                input_shape=(32, 32, 3), weights='imagenet', n_class=10,
                                optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=('accuracy'),
                                base_model_only=False):

    base_model_ = target_model(input_shape=input_shape, include_top=False, weights=weights)

    if weights is not None:
        base_model_.trainable = False

    if base_model_only:
        return base_model_

    try:
        model_ = tf.keras.Sequential(base_model_.layers + [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_class, activation='softmax')
        ])
    except ValueError:
        model_ = tf.keras.Sequential([base_model_,
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(n_class, activation='softmax')
        ])

    model_.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model_, base_model_
