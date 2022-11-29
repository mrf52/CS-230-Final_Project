import tensorflow as tf
import tensorflow.keras.layers as tfl
from preprocess import preprocess_input
from preprocess import IMG_SIZE, IMG_SHAPE
from preprocess import data_augmentation, data_augmenter


def predict(model, dataset):
    pred = model.predict(dataset)
    return pred
