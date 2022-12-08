from glob import glob
import pathlib


def get_label_counts(directory):
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
    keys = labels_to_count.keys()
    class_weights = dict()
    for key in keys:
        score = total_count / labels_to_count[key]
        class_weights[key] = score if score > 1.0 else 1.0
    return class_weights



