import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns


def plot_accuracy_loss(acc, val_acc, loss, val_loss, name='accuracy_and_loss.png'):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    # plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(name)


def plot_confusion_matrix(dataset, model, name='confusion_matrix.png'):
    gt_labels = np.concatenate([y for x, y in dataset], axis=0)
    gt_labels = gt_labels.argmax(axis=1)
    predicted_labels = model.predict(dataset).argmax(axis=1)
    confusion = tf.math.confusion_matrix(labels=gt_labels, predictions=predicted_labels, num_classes=5)
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion, annot=True, fmt="d")
    plt.savefig(name)


def evaluate_model(dataset, model):
    images = np.concatenate([x for x, y in dataset], axis=0)
    labels = np.concatenate([y for x, y in dataset], axis=0)
    model.evaluate(images, labels)

