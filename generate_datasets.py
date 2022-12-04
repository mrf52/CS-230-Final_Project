
from tensorflow.keras.preprocessing import image_dataset_from_directory


def generate_datasets(directory, batch_size=32, img_size=(160, 160)):
    train_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=img_size,
                                                 validation_split=0.2,
                                                 subset='training',
                                                 seed=42,
                                                 label_mode='categorical')
    validation_dataset = image_dataset_from_directory(directory,
                                                      shuffle=True,
                                                      batch_size=batch_size,
                                                      image_size=img_size,
                                                      validation_split=0.2,
                                                      subset='validation',
                                                      seed=42,
                                                      label_mode='categorical')
    return train_dataset, validation_dataset
