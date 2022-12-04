from train import train
from generate_datasets import generate_datasets
from preprocess import preprocess
from model import animal_model
from evaluate import plot_accuracy_loss
import argparse


def run(directory, img_size, batch_size, initial_epochs, fine_tune_epochs, base_lr):
    train_dataset, validation_dataset = generate_datasets(directory, img_size=img_size, batch_size=batch_size)
    print("Generated datasets")
    data_augmentation, preprocess_input = preprocess(train_dataset)
    model = animal_model(img_size, data_augmentation, preprocess_input)
    print("Compiled model")
    acc, val_acc, loss, val_loss = train(train_dataset, validation_dataset, model, initial_epochs=initial_epochs,
                                         fine_tune_epochs=fine_tune_epochs, base_learning_rate=base_lr)
    print("Trained model")
    plot_accuracy_loss(acc, val_acc, loss, val_loss)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', dest='directory', type=str, help='Path to directory of images')
    parser.add_argument('--initial_epochs', dest='initial_epochs', type=int, help='Number of initial epochs',
                        default=5)
    parser.add_argument('--fine_tune_epochs', dest='fine_tune_epochs', type=int, help='Number of fine tuning epochs',
                        default=5)
    parser.add_argument('--base_lr', dest='base_lr', type=float, help='Base learning rate',
                        default=0.001)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    directory = args.directory
    initial_epochs = args.initial_epochs
    fine_tune_epochs = args.fine_tune_epochs
    base_lr = args.base_lr
    img_size = (160, 160)
    batch_size = 32
    run(directory, img_size, batch_size, initial_epochs, fine_tune_epochs, base_lr)

