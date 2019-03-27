import numpy as np


def gini_from_distribution(distribution):
    """
    Calculates gini impurity for the given distribution.
    """
    sample_count = len(distribution)
    # Get the count of each unique value in distribution.
    _, per_class_count = np.unique(distribution, return_counts = True)

    # Calculate the gini impurity.
    gini_impurity = 1 - sum(np.square(per_class_count))/(sample_count * sample_count)

    return gini_impurity
    

def gini_from_frequency(frequency):
    """
    Calculates gini impurity for the given frequency of a distribution. 
    """
    sample_count = sum(frequency)
    # Calculate the gini impurity.
    gini_impurity = 1 - sum(np.square(frequency))/(sample_count * sample_count)

    return gini_impurity


def find_best_split_gini(X, Y, minleaf):
    """
    Finds the feature and the value of the feature which produces the best
    split (into 2 groups) of X and Y. The best split is the one with minimum
    gini impurity.

    Parameters:
    X: Feature matrix of the data sample. 
       Expected shape: n_samples x n_features
    Y: Corresponding labels of the data samles.
       Expected shape: n_samples x 1
    minleaf: minimum number of samples required in either of the two
             split groups.

    Returns:
    best_ftr_idx : index of the feature which produces best split.
    best_cut_val : value of the feature which produces the best split.
    best_impurity: The gini impurity for the best split.
    """

    best_ftr_idx = 0
    best_cut_val = 0
    best_impurity = 100

    # Consider each feature separately and find the one that produces the best split.
    for ftr_idx in range(X.shape[1]):
        # Find the best cut value and impurity for this feature.
        cut_val, impurity = find_best_cut_value(X[:, ftr_idx], Y, minleaf)

        if (impurity < best_impurity):
            best_cut_val = cut_val
            best_ftr_idx = ftr_idx
            best_impurity = impurity

    return best_ftr_idx, best_cut_val, best_impurity


def gini_boundary(X, Y, minleaf = 1):
    """
    Helper function to find the cut value within X that produces minimum
    gini impurity. Note that here X contains only the values of one feature.
    """
    cut_value = 0
    impurity  = 100
    M         = len(Y)
    Labels, frequency = np.unique(Y, return_counts=True)    

    # Initially assume all the points are in the right child.
    frequency_right   = dict(zip(Labels, frequency)) 
    frequency_left    = {}

    pre_gini  = gini_from_distribution(Y)

    sorted_values   = np.sort(X)
    sorted_indices  = np.argsort(X)
    sorted_labels   = Y[sorted_indices]

    j = minleaf # Discuss Index Issues

    while(j<=M-minleaf-1):
        labels                      = sorted_labels[j:]
        labels                      = labels[sorted_values[j:] == sorted_values[j]]
        labels_, counts     	    = np.unique(labels, return_counts = True)
        
        # Transfer all these points from the right child to the left. 
        # We need to update the frequency of the labels in both nodes.
        for i in range(len(labels_)):
            frequency_right[labels_[i]] = frequency_right.get(labels_[i], 0) - counts[i]
            frequency_left[labels_[i]]  = frequency_left.get(labels_[i], 0) + counts[i]

        j += len(labels)

        if j <= M-minleaf:
            gl        = gini_from_frequency(list(frequency_left.values()))
            gr        = gini_from_frequency(list(frequency_right.values()))
            post_gini = (j-1)*gl/M + (M-j+1)*gr/M

        else:
            post_gini = pre_gini + 1

        if post_gini < pre_gini:
            pre_gini  = post_gini
            cut_value = 0.5*(sorted_values[j] + sorted_values[j-1])
            impurity  = post_gini

    return cut_value, impurity
