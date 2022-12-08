import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns


def plot_accuracy_loss(train_acc, val_acc, train_loss, val_loss, name='accuracy_and_loss.png'):
    """
    Generates a plot of accuracy and loss values during training.

    :param train_acc: Training accuracy values.
    :param val_acc: Validation accuracy values.
    :param train_loss: Training loss values.
    :param val_loss: Validation loss values.
    :param name: Name to save image file to.
    """
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(name)


def plot_confusion_matrix(dataset, model, name='confusion_matrix.png'):
    """
    Generates a confusion matrix for a specified dataset.

    :param dataset: tf.data.Dataset of labeled input images.
    :param model: tf.keras.model to predict with
    :param name: Name to save image file to.
    """
    gt_labels = np.concatenate([y for x, y in dataset], axis=0)
    gt_labels = gt_labels.argmax(axis=1)
    predicted_labels = model.predict(dataset).argmax(axis=1)
    confusion = tf.math.confusion_matrix(labels=gt_labels, predictions=predicted_labels, num_classes=5)
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap=sns.cm.rocket_r)
    plt.savefig(name)


def plot_confusion_matrix_normalized(dataset, model, name='confusion_matrix_norm.png'):
    """
    Generates a normalized confusion matrix for a specified dataset.

    :param dataset: tf.data.Dataset of labeled input images.
    :param model: tf.keras.model to predict with
    :param name: Name to save image file to.
    """
    gt_labels = np.concatenate([y for x, y in dataset], axis=0)
    gt_labels = gt_labels.argmax(axis=1)
    predicted_labels = model.predict(dataset).argmax(axis=1)
    confusion = tf.math.confusion_matrix(labels=gt_labels, predictions=predicted_labels, num_classes=5)
    confusion = confusion.numpy().astype(float)
    row_total = np.sum(confusion, axis=1)
    for row in range(5):
        for col in range(5):
            confusion[row, col] = confusion[row, col] / row_total[row]
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion, annot=True, cmap=sns.cm.rocket_r)
    plt.savefig(name)


def evaluate_model(dataset, model):
    """
    Evaluates the model on a specified dataset.

    :param dataset: tf.data.Dataset of labeled input images.
    :param model: tf.keras.model to predict with
    """
    images = np.concatenate([x for x, y in dataset], axis=0)
    labels = np.concatenate([y for x, y in dataset], axis=0)
    model.evaluate(images, labels)

