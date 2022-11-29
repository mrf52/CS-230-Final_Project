import tensorflow as tf
import tensorflow.keras.layers as tfl
from preprocess import preprocess_input
from preprocess import IMG_SIZE, IMG_SHAPE
from preprocess import data_augmentation, data_augmenter


def animal_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    '''

    input_shape = image_shape + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')

    # freeze the base model by making it non trainable
    base_model.trainable = False
    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape)
    # apply data augmentation to the inputs
    x = data_augmentation(inputs)

    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)

    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False)

    # add new top layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x)
    # include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)
    # use a prediction layer with 5 neurons (1 per class)
    outputs = tfl.Dense(5)(x)
    outputs = tfl.Activation('softmax')(outputs)

    model = tf.keras.Model(inputs, outputs)

    return model


def train(train_dataset, validation_dataset, initial_epochs=5,
          fine_tune_epochs=5, base_learning_rate=0.001):
    # compile the model
    model = animal_model(IMG_SIZE, data_augmentation)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # train initial model
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)
    # save accuracy and loss
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # fine tune
    base_model = model.layers[4]
    base_model.trainable = True
    fine_tune_at = 120
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.1 * base_learning_rate),
                  metrics=['accuracy'])
    # train fine-tuning epochs
    total_epochs = initial_epochs + fine_tune_epochs
    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset)
    # save accuracy and loss
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']
    return acc, val_acc, loss, val_loss



