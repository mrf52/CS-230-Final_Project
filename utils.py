from glob import glob
import pathlib


def get_label_counts(directory):
    """
    Counts number of images belonging to each species class.

    :param directory: Directory of input images. Organized into subdirectories named by species.
    :return: labels_to_count: Dictionary mapping class to number of corresponding images.
    :return: total_count: Total number of input images.
    """
    directory = pathlib.Path(directory)
    total_count = len(list(directory.glob('*/*.jpg')))
    labels_to_count = dict()
    labels_to_count[0] = len(list(directory.glob('bobcat/*.jpg')))
    labels_to_count[1] = len(list(directory.glob('cat/*.jpg')))
    labels_to_count[2] = len(list(directory.glob('coyote/*.jpg')))
    labels_to_count[3] = len(list(directory.glob('dog/*.jpg')))
    labels_to_count[4] = len(list(directory.glob('fox/*.jpg')))
    return labels_to_count, total_count


def create_class_weight(labels_to_count, total_count):
    """
    Calculates class weight for each species class (how much to oversample by during training).

    :param labels_to_count: Dictionary mapping class to number of corresponding images.
    :param total_count: Total number of input images.
    :return: class_weights: Dictionary mapping class to class weight.
    """
    keys = labels_to_count.keys()
    class_weights = dict()
    for key in keys:
        score = total_count / labels_to_count[key]
        class_weights[key] = score if score > 1.0 else 1.0
    return class_weights



