import tensorflow as tf


def train(train_dataset, validation_dataset, model, initial_epochs=5,
          fine_tune_epochs=5, base_learning_rate=0.001):
    # compile the model
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



