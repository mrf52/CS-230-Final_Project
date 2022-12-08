
from tensorflow.keras.preprocessing import image_dataset_from_directory


def generate_datasets(directory, test_directory,  batch_size=32, img_size=(160, 160), validation_split=0.2):
    """
    Splits input dataset into train, validation, test components.

    :param directory: Directory of input images. Organized into subdirectories named by species.
    :param test_directory: Directory of test input images. Organized into subdirectories named by species.
    :param batch_size: Batch size. Default 32.
    :param img_size: Size of input images. Default (160,160)
    :param validation_split: Fraction of dataset to reserve for validation.
    :return: train_dataset: Dataset of training samples.
    :return: validation_dataset: Dataset of validation samples.
    :return: test_dataset: Dataset of testing samples.
    """
    test_dataset = image_dataset_from_directory(test_directory,
                                                shuffle=True,
                                                batch_size=batch_size,
                                                image_size=img_size,
                                                seed=42,
                                                label_mode='categorical'
                                                )
    train_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=img_size,
                                                 validation_split=validation_split,
                                                 subset='training',
                                                 seed=42,
                                                 label_mode='categorical')
    validation_dataset = image_dataset_from_directory(directory,
                                                      shuffle=True,
                                                      batch_size=batch_size,
                                                      image_size=img_size,
                                                      validation_split=validation_split,
                                                      subset='validation',
                                                      seed=42,
                                                      label_mode='categorical')
    return train_dataset, validation_dataset, test_dataset
