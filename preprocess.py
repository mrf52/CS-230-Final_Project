import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip


def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))

    return data_augmentation


def preprocess(train_dataset):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    data_augmentation = data_augmenter()
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    return data_augmentation, preprocess_input




