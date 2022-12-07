from train import train
from generate_datasets import generate_datasets
from preprocess import preprocess
from model import animal_model
from evaluate import plot_accuracy_loss, plot_confusion_matrix, evaluate_model
from utils import get_label_counts, create_class_weight
import argparse


def run(directory, test_directory, img_size, batch_size, initial_epochs, fine_tune_epochs, base_lr,
        validation_split, balance):
    train_dataset, validation_dataset, test_dataset = generate_datasets(directory, test_directory,
                                                                        img_size=img_size, batch_size=batch_size,
                                                                        validation_split=validation_split)
    print("Generated datasets")
    if balance:
        labels_to_count, total_count = get_label_counts(directory)
        class_weights = create_class_weight(labels_to_count, total_count)
        print("Calculated class weights")
    else:
        class_weights = None
    data_augmentation, preprocess_input = preprocess(train_dataset)
    model = animal_model(img_size, data_augmentation, preprocess_input)
    print("Compiled model")
    acc, val_acc, loss, val_loss = train(train_dataset, validation_dataset, model, initial_epochs=initial_epochs,
                                         fine_tune_epochs=fine_tune_epochs, base_learning_rate=base_lr,
                                         class_weights=class_weights)
    print("Trained model")
    plot_accuracy_loss(acc, val_acc, loss, val_loss)
    plot_confusion_matrix(test_dataset, model, name='confusion_matrix_test.png')  # confusion for test dataset
    plot_confusion_matrix(train_dataset, model, name='confusion_matrix_train.png')  # confusion for train dataset
    plot_confusion_matrix(validation_dataset, model, name='confusion_matrix_val.png')  # confusion for val dataset
    print("Evaluated model")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', dest='directory', type=str, help='Path to directory of images')
    parser.add_argument('--test_directory', dest='test_directory', type=str, help='Path to directory of test images')
    parser.add_argument('--initial_epochs', dest='initial_epochs', type=int, help='Number of initial epochs',
                        default=5)
    parser.add_argument('--fine_tune_epochs', dest='fine_tune_epochs', type=int, help='Number of fine tuning epochs',
                        default=5)
    parser.add_argument('--base_lr', dest='base_lr', type=float, help='Base learning rate',
                        default=0.001)
    parser.add_argument('--validation_split', dest='validation_split', type=float,
                        help='Fraction of data reserved for validation',
                        default=0.2)
    parser.add_argument('--balance', dest='balance', type=bool,
                        help='Whether to balance dataset with weights during training',
                        default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    directory = args.directory
    test_directory = args.test_directory
    initial_epochs = args.initial_epochs
    fine_tune_epochs = args.fine_tune_epochs
    base_lr = args.base_lr
    img_size = (160, 160)
    batch_size = 32
    validation_split = args.validation_split
    balance = args.balance
    run(directory, test_directory, img_size, batch_size, initial_epochs, fine_tune_epochs, base_lr, validation_split,
        balance)

