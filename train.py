import tensorflow as tf


def train(train_dataset, validation_dataset, model, initial_epochs=5,
          fine_tune_epochs=5, base_learning_rate=0.001, class_weights=None):
    """
    Performs training on animal species classifier.

    :param train_dataset: Dataset of training samples
    :param validation_dataset: Dataset of validation samples
    :param model: tf.keras.model to train with
    :param initial_epochs: Number of baseline model epochs
    :param fine_tune_epochs: Number of fine-tuning epochs
    :param base_learning_rate: Learning rate for the baseline epochs
    :param class_weights: Optional, dictionary of values to oversample classes by
    :return: Accuracy and loss of the training and validation datasets
    """
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # train initial model
    if class_weights is None:
        history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)
    else:
        history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs,
                            class_weight=class_weights)
    # save accuracy and loss
    train_acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']
    train_loss = history.history['loss']
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
    if class_weights is None:
        history_fine_tune = model.fit(train_dataset,
                                      epochs=total_epochs,
                                      initial_epoch=history.epoch[-1],
                                      validation_data=validation_dataset)
    else:
        history_fine_tune = model.fit(train_dataset,
                                      epochs=total_epochs,
                                      initial_epoch=history.epoch[-1],
                                      validation_data=validation_dataset,
                                      class_weight=class_weights)
    # save accuracy and loss
    train_acc += history_fine_tune.history['accuracy']
    val_acc += history_fine_tune.history['val_accuracy']

    train_loss += history_fine_tune.history['loss']
    val_loss += history_fine_tune.history['val_loss']
    return train_acc, val_acc, train_loss, val_loss



