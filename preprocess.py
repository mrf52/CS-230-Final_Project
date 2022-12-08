import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip


def data_augmenter():
    """
    Creates a data augmentation function that applies a random horizontal flip.

    :return: tf.keras.Sequential
    """
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))

    return data_augmentation


def preprocess(train_dataset):
    """
    Preprocessing step of training.

    :param train_dataset: Dataset of training samples
    :return: Data augmentation function, preprocessing function
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    data_augmentation = data_augmenter()
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    return data_augmentation, preprocess_input




