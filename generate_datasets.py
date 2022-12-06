
from tensorflow.keras.preprocessing import image_dataset_from_directory


def generate_datasets(directory, test_directory,  batch_size=32, img_size=(160, 160), validation_split=0.2):
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
