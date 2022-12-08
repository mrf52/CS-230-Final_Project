import tensorflow as tf
import tensorflow.keras.layers as tfl


def animal_model(img_size, data_augmentation, preprocess_input):
    """
    Compiles a categorical classification model for classifying animal species.

    :param img_size: Size of the input images
    :param data_augmentation: Defined data augmentation function
    :param preprocess_input: Defined image preprocessing function
    :return: tf.keras.model to use for training
    """
    input_shape = img_size + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    # Apply data augmentation and preprocessing
    x = data_augmentation(inputs)
    x = preprocess_input(x)

    # Base model: train only top layers
    x = base_model(x, training=False)
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dropout(0.2)(x)
    outputs = tfl.Dense(5)(x)
    outputs = tfl.Activation('softmax')(outputs)

    model = tf.keras.Model(inputs, outputs)

    return model
