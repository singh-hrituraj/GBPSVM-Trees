import numpy as np


def gini_impurity(distribution):
    """
    Calculates gini impurity for the given distribution
    """
    sample_count = length(distribution)
    # Get the count of each unique value in distribution.
    _, per_class_count = np.unique(distribution, return_counts = True)

    # Calculate the gini impurity.
    gini_impurity = 1 - sum(np.square(per_class_count))/(sample_count * sample_count)

    return gini_impurity




