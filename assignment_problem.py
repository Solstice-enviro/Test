"""
This simple script solves the "assignment problem", namely identifying the
best way of splitting the six rectangular regions into three datasets.
"""
import numpy as np
from itertools import combinations


def generate_all_assignment_indices():
    """
    This function generates all assignment index triplets,
    assuming there are 6 sub-regions. The first 3 go to the
    training set, the next 2 go to the testing set and the final goes to the validation set.

    For example, an "assignment" is of the form [[2, 3, 4], [0, 5], [1]]
    """
    number_of_regions = 6

    list_assignments = []

    indices = np.arange(number_of_regions)

    list_training_indices = list(combinations(indices, 3))

    for training_indices in list_training_indices:

        left_over_indices = list(set(indices).difference(set(training_indices)))

        list_test_indices = list(combinations(left_over_indices, 2))

        for test_indices in list_test_indices:

            valid_index = list(set(left_over_indices).difference(set(test_indices)))
            assignment = [list(training_indices), list(test_indices), list(valid_index)]
            list_assignments.append(assignment)

    return list_assignments


def compute_class_ratios(list_regions):
    """
    This function takes in a list of arrays containing the
    areas of each class and returns the ratio of each class
    for the sum of all regions.
    """

    areas = np.array([0., 0., 0., 0.])

    for region in list_regions:
        areas = areas + region

    total_area = np.sum(areas)

    class_ratios = areas/total_area

    return class_ratios


def get_ratios(assignment_indices, list_regions):
    """
    This utility function simply computes the ratios for the training, testing and validation
    regions, based on the assigment indices.
    """

    train_indices, test_indices, validation_indices = assignment_indices

    train_regions = [list_regions[i] for i in train_indices]
    train_ratio = compute_class_ratios(train_regions)

    test_regions = [list_regions[i] for i in test_indices]
    test_ratio = compute_class_ratios(test_regions)

    valid_regions = [list_regions[i] for i in validation_indices]
    valid_ratio = compute_class_ratios(valid_regions)

    return train_ratio, test_ratio, valid_ratio


def print_assignment(list_region_labels, dataset_indices, ratios):
    """
    Convenience function to output results to screen.
    """
    list_classes = ["Marsh", "Open Water", "Swamp", "Upland"]
    print("     assignment: " +" ".join([list_region_labels[i] for i in dataset_indices]))
    print("     class ratios:")
    for class_name, r in zip(list_classes, ratios):
        print(f"        {class_name}: {100 * r:3.4f}%")


def get_metric(train_ratio, test_ratio, valid_ratio):
    """
    This computes a made up "distance" between the regions for the purpose
    of finding the best one.
    """

    # since Swamp is so rare, we'll over weight it
    weights = np.array([1., 1., 10**3, 1.])
    d1 = np.linalg.norm(weights*(train_ratio - test_ratio))
    d2 = np.linalg.norm(weights*(train_ratio - valid_ratio))
    metric = np.sqrt(d1**2 + d2**2)
    return metric


# number of hectares of each class for the rectangular regions, according to the report.
# The class order is "Marsh", "Open Water", "Swamp", "Upland"
train1 = np.array([739.1, 374.8, 3.6, 18293.2])
train2 = np.array([2368.9, 1975.0, 26.8, 34289.8])
train3 = np.array([1222.1, 179.8, 0.9, 27712.4])
test1 = np.array([669.6, 161.0, 8.7, 13423.9])
test2 = np.array([2012.3, 2619.8, 0.3, 21794.9])
validate = np.array([2064.1, 1000.8, 14.9, 32239.9])


list_regions = [train1, train2, train3, test1, test2, validate]
list_region_labels = ['train-1', 'train-2', 'train-3', 'test-1', 'test-2', 'valid-1']

if __name__ == '__main__':

    list_assignments = generate_all_assignment_indices()
    list_metrics = []

    for assignment_indices in list_assignments:
        train_ratio, test_ratio, valid_ratio = get_ratios(assignment_indices, list_regions)
        metric = get_metric(train_ratio, test_ratio, valid_ratio)
        list_metrics.append(metric)

    smallest_metric_index = np.argmin(list_metrics)
    best_assignment_indices = list_assignments[smallest_metric_index]

    train_ratio, test_ratio, valid_ratio = get_ratios(best_assignment_indices, list_regions)

    print("The best assignment of regions is given by")
    print("====================================================")
    print("     training set:")
    print_assignment(list_region_labels, best_assignment_indices[0], train_ratio)
    print("====================================================")
    print("     test set    :")
    print_assignment(list_region_labels, best_assignment_indices[1], test_ratio)
    print("====================================================")
    print("     validation  set:")
    print_assignment(list_region_labels, best_assignment_indices[2], valid_ratio)
